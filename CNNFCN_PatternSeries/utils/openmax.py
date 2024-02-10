import numpy as np
import scipy.spatial.distance as spd
import scipy as sp
import libmr
from sklearn import metrics

from sklearn.metrics import roc_auc_score


def calc_auroc(id_test_results, ood_test_results):
    # calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))

    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result


def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    n_classes = openmax_fc8.shape[0]
    scores = []
    for category in range(n_classes):
        scores += [sp.exp(openmax_fc8[category])]

    total_denominator = sp.sum(sp.exp(openmax_fc8)) + sp.exp(sp.sum(openmax_score_u))
    prob_scores = scores / total_denominator
    prob_unknowns = sp.exp(sp.sum(openmax_score_u)) / total_denominator

    results = np.array(prob_scores.tolist() + [prob_unknowns])

    # modified_scores = [prob_unknowns] + prob_scores.tolist()
    # assert len(modified_scores) == (NCLASSES+1)
    # return modified_scores

    return results


def compute_distance(query_vector, mean_vec, distance_type='eucos'):
    """ 

    Output:
    --------
    query_distance : Distance between respective channels

    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec, query_vector) / 200. + spd.cosine(mean_vec, query_vector)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec, query_vector)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec, query_vector)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


import scipy.spatial.distance as spd


def compute_channel_distances(mean_vector, features):
    mean_vector = mean_vector.cpu().data.numpy()
    features = features.cpu()

    eu, cos, eu_cos = [], [], []

    for feat in features:
        feat = feat.data.numpy()

        eu.append(spd.euclidean(mean_vector, feat))
        cos.append(spd.cosine(mean_vector, feat))
        eu_cos.append(spd.euclidean(mean_vector, feat) / 200. +
                      spd.cosine(mean_vector, feat))

    eu_dist = np.array(eu)
    cos_dist = np.array(cos)
    eucos_dist = np.array(eu_cos)
    channel_distances = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean': eu_dist}
    return channel_distances


def weibull_tailfitting(means, dists, categories,
                        tailsize=20,
                        distance_type='eucos'):
    weibull_model = {}

    for mean, dist, category in zip(means, dists, categories):
        weibull_model[category] = {}

        weibull_model[category]['distances_%s' % distance_type] = dist[distance_type]
        weibull_model[category]['mean_vec'] = mean

        distance_scores = dist[distance_type].tolist()

        mr = libmr.MR()

        tailtofit = sorted(distance_scores)[-tailsize:]

        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = mr

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [weibull_model[category_name]['distances_%s' % distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def recalibrate_scores(weibull_model, input_score, num_classes, class_to_idx_dict,
                       alpharank=4, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
        Input:
            weibull_model: dict
            sample_features: NumPy array
        Output:
            openmax_probab : openmax probability and softmax probability"""
    logits = input_score[:num_classes]
    ranked_list = logits.argsort().ravel()[::-1]

    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_classes)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_layer = []
    openmax_unknown = []

    for cls_indx in range(num_classes):
        category_weibull = query_weibull(cls_indx, weibull_model, distance_type=distance_type)
        distance = compute_distance(input_score, category_weibull[0].cpu().detach().numpy(),
                                    distance_type=distance_type)

        wscore = category_weibull[2].w_score(distance)
        modified_unit = logits[cls_indx] * (1 - wscore * ranked_alpha[cls_indx])
        openmax_layer += [modified_unit]
        openmax_unknown += [logits[cls_indx] - modified_unit]

    openmax_fc8 = np.asarray(openmax_layer)
    openmax_score_u = np.asarray(openmax_unknown)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)

    ss = input_score[:num_classes+1]
    softmax_prob = softmax(np.array(ss.ravel()))
    """
    logits = [] 
    for indx in range(NCLASSES):
        logits += [sp.exp(img_features[indx])]
    den = sp.sum(sp.exp(img_features))
    softmax_probab = logits/den

    return np.asarray(openmax_probab), np.asarray(softmax_probab)
    """
    return openmax_probab, softmax_prob


def get_scores(weibull_model, input_scores, num_classes, class_to_idx_dict, open_set_class_to_idx_dict):
    """Input:
            weibull_model: dict
            input_scores: List
        Output:
            openmax_probab : """
    openmax_probs = []
    softmax_probs = []

    for ips in input_scores:
        sample_features = ips.cpu().detach().numpy()
        openmax, softmax = recalibrate_scores(weibull_model, sample_features, num_classes, class_to_idx_dict, alpharank=num_classes)
        openmax_probs.append(openmax)
        softmax_probs.append(softmax)

    return np.array(openmax_probs), np.array(softmax_probs)


def get_avg_prec_recall(ConfMatrix, existing_class_names, excluded_classes=None):
    """Get average recall and precision, using class frequencies as weights, optionally excluding
    specified classes"""

    class2ind = dict(zip(existing_class_names, range(len(existing_class_names))))
    included_c = np.full(len(existing_class_names), 1, dtype=bool)

    if not (excluded_classes is None):
        excl_ind = [class2ind[excl_class] for excl_class in excluded_classes]
        included_c[excl_ind] = False

    pred_per_class = np.sum(ConfMatrix, axis=0)
    nonzero_pred = (pred_per_class > 0)

    included = included_c & nonzero_pred
    support = np.sum(ConfMatrix, axis=1)
    weights = support[included] / np.sum(support[included])

    prec = np.diag(ConfMatrix[included, :][:, included]) / pred_per_class[included]
    prec_avg = np.dot(weights, prec)

    # rec = np.diag(ConfMatrix[included_c,:][:,included_c])/support[included_c]
    rec_avg = np.trace(ConfMatrix[included_c, :][:, included_c]) / np.sum(support[included_c])

    return prec_avg, rec_avg


def analyze_classification(y_pred, y_true, class_names):
    """
    For an array of label predictions and the respective true labels, shows confusion matrix, accuracy, recall, precision etc:
    Input:
        y_pred: 1D array of predicted labels (class indices)
        y_true: 1D array of true labels (class indices)
        class_names: 1D array or list of class names in the order of class indices.
            Could also be integers [0, 1, ..., num_classes-1].
        excluded_classes: list of classes to be excluded from average precision, recall calculation (e.g. OTHER)
    """

    # Trim class_names to include only classes existing in y_pred OR y_true
    in_pred_labels = set(list(y_pred))
    in_true_labels = set(list(y_true))

    existing_class_ind = sorted(list(in_pred_labels | in_true_labels))
    class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
    existing_class_names = [class_strings[ind][:min(35, len(class_strings[ind]))] for ind in
                            existing_class_ind]  # a little inefficient but inconsequential

    # Confusion matrix
    ConfMatrix = metrics.confusion_matrix(y_true, y_pred)

    # if self.print_conf_mat:
    #     print_confusion_matrix(ConfMatrix, label_strings=self.existing_class_names, title='Confusion matrix')
    #     print('\n')
    # if self.plot:
    #     plt.figure()
    #     plot_confusion_matrix(ConfMatrix, self.existing_class_names)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    ConfMatrix_normalized_row = ConfMatrix.astype('float') / ConfMatrix.sum(axis=1)[:, np.newaxis]

    # if self.print_conf_mat:
    #     print_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
    #                            title='Confusion matrix normalized by row')
    #     print('\n')
    # if self.plot:
    #     plt.figure()
    #     plot_confusion_matrix(self.ConfMatrix_normalized_row, label_strings=self.existing_class_names,
    #                           title='Confusion matrix normalized by row')
    #
    #     plt.show(block=False)

    # Analyze results
    total_accuracy = np.trace(ConfMatrix) / len(y_true)
    print('Overall accuracy: {:.3f}\n'.format(total_accuracy))

    # returns metrics for each class, in the same order as existing_class_names
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, labels=existing_class_ind)
    #
    ##### Add a new metric F1/4 = (beta^2 + 1)*P*R / (beta^2*P + R)
    # 20231003
    f1_4 = (17 * precision * recall) / (precision + 16 * recall)

    # Print report
    # print(self.generate_classification_report())

    # Calculate average precision and recall
    prec_avg, rec_avg = get_avg_prec_recall(ConfMatrix, existing_class_names)
    # if excluded_classes:
    #     print(
    #         "\nAverage PRECISION: {:.2f}\n(using class frequencies as weights, excluding classes with no predictions and predictions in '{}')".format(
    #             self.prec_avg, ', '.join(excluded_classes)))
    #     print(
    #         "\nAverage RECALL (= ACCURACY): {:.2f}\n(using class frequencies as weights, excluding classes in '{}')".format(
    #             self.rec_avg, ', '.join(excluded_classes)))

    # Make a histogram with the distribution of classes with respect to precision and recall
    # self.prec_rec_histogram(self.precision, self.recall)

    return {"total_accuracy": total_accuracy, "precision": precision, "recall": recall,
            "f1": f1, "f1/4": f1_4, "support": support, "prec_avg": prec_avg,
            "rec_avg": rec_avg}
