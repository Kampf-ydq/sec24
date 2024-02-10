import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from fitter import Fitter
from matplotlib import rcParams

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
def plot_prob_function(predictions):
    # x = torch.max(predictions, dim=1)[0]

    probs = torch.nn.functional.softmax(predictions)  # (400, 5)
    max_probs = torch.max(probs, dim=1)[0]  # Softmax layer
    lst = max_probs.tolist()
    exist_confid_score = lst[:100]
    unknw_confid_score = lst[800:900]
    # fitted distribution
    logger.info("[*] Math Analysis ...")
    # logger.info("Fit {} / All Samples: {}".format(200, len(lst)))
    from fitter import Fitter
    import scipy.stats as st
    exist_f = Fitter(exist_confid_score)
    exist_f.fit()
    best_dis = list(exist_f.get_best().keys())[0]
    logger.info("The best distribution of [Existing] is: {}".format(exist_f.get_best()))

    unknw_f = Fitter(unknw_confid_score)
    unknw_f.fit()
    logger.info("The best distribution of [Unknown] is: {}".format(unknw_f.get_best()))

    # build the class of best distribution
    # params = exist_f.get_best()[best_dis]
    # X = st.mielke(k=params["k"], s=params["s"], loc=params["loc"], scale=params["scale"])
    # CDF and PDF.
    # cdf_val = X.cdf(confid_score)
    # pdf_val = X.pdf(confid_score)

    # plt.bar(confid_score, normalize(pdf_val))
    exist_f.hist()
    unknw_f.hist()
    plt.show()



def normalize(data):
    d = np.array(data)
    min_val = np.min(d)
    max_val = np.max(d)
    noral = (d - min_val) / (max_val - min_val)
    return noral.tolist()

def test_gamma():
    from scipy import stats
    data = stats.gamma.rvs(2, loc=1.5, scale=2, size=10000)

    f = Fitter(data, bins=100)
    f.xmin = -10  # should have no effect
    f.xmax = 1000000  # no effect
    f.xmin = 0.1
    f.xmax = 10
    f.distributions = ['gamma', "alpha"]
    f.fit()
    df = f.summary()
    assert len(df)

    f.plot_pdf(names=["gamma"])
    f.plot_pdf(names="gamma")

    res = f.get_best()
    assert "gamma" in res.keys()


def test_others():
    from scipy import stats
    data = stats.gamma.rvs(2, loc=1.5, scale=2, size=1000)
    f = Fitter(data, bins=100, distributions="common")
    f.fit()
    assert f.df_errors.loc["gamma"].loc['aic'] > 100

    f = Fitter(data, bins=100, distributions="gamma")
    f.fit()
    assert f.df_errors.loc["gamma"].loc['aic'] > 100


def test_n_jobs_api():
    from scipy import stats
    data = stats.gamma.rvs(2, loc=1.5, scale=2, size=1000)
    f = Fitter(data, distributions="common")
    f.fit(n_jobs=-1)
    f.fit(n_jobs=1)

if __name__ == '__main__':
    # Confidence score: confidence score of field categorization after different segmenter divisions
    tshark_score = [1.0, 0.9999942779541016, 1.0, 1.0, 1.0, 0.9049798250198364, 0.9551929831504822, 0.9993504881858826,
                   0.9999935626983643, 0.9999997615814209, 1.0, 0.9980826377868652, 1.0, 1.0, 1.0, 1.0,
                   0.9975132942199707, 1.0, 1.0, 1.0, 0.9809656143188477, 0.9747959971427917, 0.9910916686058044,
                   0.9999973773956299, 1.0, 1.0, 0.9739266037940979, 1.0, 1.0, 1.0, 1.0, 0.9986554384231567, 1.0, 1.0,
                   1.0, 0.9962947964668274, 0.8919768333435059, 0.9996216297149658, 0.9999963045120239, 1.0, 1.0,
                   0.9980826377868652, 1.0, 1.0, 1.0, 1.0, 0.9986554384231567, 1.0, 1.0, 1.0, 1.0, 0.9985221028327942,
                   0.9999973773956299, 1.0, 1.0, 1.0, 0.9999909400939941, 1.0, 1.0, 1.0, 0.9999780654907227,
                   0.9806868433952332, 0.9999991655349731, 0.9998637437820435, 1.0, 1.0, 0.7185987830162048, 1.0, 1.0,
                   1.0, 0.9999926090240479, 0.981155514717102, 0.9716185927391052, 0.9768626689910889, 1.0, 1.0,
                   0.9999953508377075, 1.0, 1.0, 1.0, 1.0, 0.9999958276748657, 1.0, 1.0, 1.0, 0.9999969005584717,
                   0.9817051887512207, 1.0, 1.0, 1.0, 1.0, 0.998012900352478, 0.9998598098754883, 1.0, 1.0, 1.0,
                   0.9881812930107117, 0.9999990463256836, 1.0, 1.0]
    fixed_bytes_2 = [0.9999991655349731, 1.0, 0.9999991655349731, 0.64665287733078, 0.9999998807907104, 0.9839703440666199, 0.9999998807907104, 0.9997565150260925, 0.9999998807907104, 0.9989681243896484, 1.0, 0.9999995231628418, 1.0, 1.0, 1.0, 0.88557368516922, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.999991774559021, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.9999982118606567, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.8021253943443298, 1.0, 0.8431448340415955, 1.0, 1.0, 1.0, 0.6153939962387085, 1.0, 0.9999951124191284, 1.0, 1.0, 1.0, 0.6153939962387085, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6153939962387085, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6153939962387085, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6153939962387085, 1.0, 1.0]
    netzob_score = [0.9999983310699463, 1.0, 0.9969140291213989, 0.9932043552398682, 0.9955293536186218, 0.9999967813491821, 0.9932043552398682, 0.9999945163726807, 1.0, 0.996507465839386, 0.9999958276748657, 0.999636173248291, 0.9790971875190735, 0.999636173248291, 1.0, 1.0, 0.9848296046257019, 0.9688392877578735, 0.9790971875190735, 0.9999819993972778, 0.9999983310699463, 0.6474360823631287, 0.9336679577827454, 0.9932043552398682, 0.9999996423721313, 0.9932043552398682, 0.9336679577827454, 0.6868436932563782, 1.0, 0.9987021684646606, 1.0, 0.531281590461731, 1.0, 0.531281590461731, 0.9932043552398682, 0.9955293536186218, 0.8324613571166992, 0.9999940395355225, 0.996507465839386, 0.6868436932563782, 0.9848296046257019, 0.999636173248291, 0.996507465839386, 1.0, 0.9999940395355225, 1.0, 1.0, 0.9848296046257019, 1.0, 1.0, 0.9999978542327881, 0.9790971875190735, 0.9999967813491821, 1.0, 0.9999998807907104, 1.0, 0.991134524345398, 0.9999967813491821, 0.9790971875190735, 0.9932043552398682, 0.9790971875190735, 0.9790971875190735, 0.9635410904884338, 1.0, 1.0, 0.9848296046257019, 0.9790971875190735, 0.996507465839386, 0.9999967813491821, 1.0, 0.996507465839386, 0.9999983310699463, 0.999636173248291, 0.9883576035499573, 0.999636173248291, 1.0, 0.9790971875190735, 1.0, 1.0, 0.9436741471290588, 0.9932043552398682, 0.9790971875190735, 0.6474360823631287, 0.6421467065811157, 0.5865006446838379, 0.9999940395355225, 1.0, 1.0, 0.9436741471290588, 1.0, 0.9932043552398682, 1.0, 0.999636173248291, 1.0, 0.999636173248291, 0.9932043552398682, 1.0, 0.9999983310699463, 0.9999940395355225, 1.0]

    # Unit is length
    flow_lgths = [6, 890, 3, 216, 328, 160, 526, 86, 1904, 12, 394, 548, 110, 1, 1345, 2, 344, 4, 64, 1174, 4, 834, 1006, 10, 20,
     292, 228, 370, 70, 16, 20, 10, 20, 194, 376, 70, 152, 94, 3190, 530, 228, 264, 150, 36, 150, 74, 12, 120, 28, 68,
     106, 506, 14, 268, 14, 66, 482, 48, 8, 184, 34, 84, 16, 28, 38, 74, 32, 68, 30, 8, 10, 30, 46, 8, 48, 20, 4, 18,
     304, 116, 738, 302, 284, 88, 436, 52, 114, 208, 40, 78, 940, 72, 62, 522, 400, 2, 136, 74, 8, 2, 100, 2, 10, 42,
     46, 66, 20, 66, 22, 18, 204, 164, 178, 152, 6, 34, 238, 158, 252, 934, 552, 154, 68, 34, 34, 8, 246, 120, 660, 400,
     14, 542, 236, 20, 114, 34, 2, 26, 18, 38, 10, 98, 40, 26, 80, 94, 178, 80, 68, 142, 320, 166, 148, 390, 686, 270,
     94, 576, 116, 28, 10, 6, 6, 8, 192, 62, 592, 6, 10, 122, 778, 196, 88, 558, 252, 350, 666, 644, 118, 498, 178, 6,
     4, 34, 4, 86, 340, 186, 392, 1972, 172, 334, 86, 210, 340, 114, 164, 68, 354, 52, 6, 12, 16, 294, 64, 38, 4, 4, 6,
     160, 252, 154, 542, 186, 728, 250, 164, 258, 402]
    short_flow = [sf for sf in flow_lgths if sf <= 500]
    long_flow = [lf for lf in flow_lgths if lf > 500]
    # Unit is second (/s)
    all_time = [2.03, 444.328, 11.427, 107.111, 163.134, 79.099, 262.124, 42.067, 951.707, 618.176, 196.133, 273.244, 49.021, 0, 671.831, 0.01, 171.071, 3.762, 31.057, 586.258, 0.213, 416.326, 502.381, 55.354, 8.996, 145.162, 113.063, 184.178, 34.024, 7.011, 9.017, 4.002, 8.959, 96.054, 187.206, 34.023, 75.078, 46.024, 1595.067, 264.205, 113.042, 131.076, 74.033, 17.022, 74.043, 36.044, 5.011, 59.052, 13.033, 33.043, 52.032, 252.151, 6.011, 133.122, 6.033, 32.022, 240.135, 22.999, 2.973, 91.109, 16.005, 41.024, 7.012, 13.017, 18.026, 36.024, 14.984, 33.04, 13.981, 3.018, 4.015, 14.028, 22.03, 2.997, 23.02, 8.997, 1, 7.987, 151.004, 57.045, 368.103, 150.073, 141.056, 43.046, 217.084, 25.064, 56.054, 103.075, 19.022, 38.018, 469.322, 35.032, 30.022, 260.217, 199.099, 0.011, 67.04, 36.011, 2.993, 0.012, 48.997, 0.012, 3.996, 20.046, 21.973, 32.006, 9.008, 32.03, 10.008, 8.021, 101.085, 81.062, 88.077, 75.057, 2.012, 16.021, 118.077, 78.042, 125.1, 466.263, 275.141, 76.088, 33.033, 16.022, 16.02, 3.011, 122.088, 59.076, 329.114, 199.101, 6.023, 270.138, 117.055, 8.991, 56.05, 16, 0.012, 12.006, 7.974, 18.04, 4.007, 48.013, 18.992, 12.005, 39.027, 46.02, 88.048, 39.045, 33.023, 70.068, 159.076, 82.055, 73.045, 194.084, 342.272, 134.153, 46.031, 287.16, 57.053, 13.012, 4.021, 2.012, 2.011, 3.014, 95.051, 30.045, 295.166, 2.012, 4.011, 60.021, 388.235, 97.088, 43.034, 278.204, 125.087, 174.208, 332.296, 321.247, 58.07, 248.218, 88.077, 1.996, 1.002, 15.974, 0.995, 41.992, 169.117, 92.054, 195.052, 985.538, 85.033, 166.147, 42.056, 104.098, 169.053, 56.045, 81.084, 33.034, 176.078, 25.021, 1.996, 5.009, 6.998, 146.151, 31.016, 18.004, 0.987, 0.988, 1.993, 79.005, 125.097, 76.099, 270.162, 92.077, 363.083, 124.1, 81.051, 128.073, 200.176]
    all_time = [aft for aft in all_time if aft >= 1]
    short_time = [sft for sft in all_time if sft <= 200]
    long_time = [lft for lft in all_time if lft > 200 and lft <= 400]

    #----------------- pdf ------------------#
    # n, bis, p = plt.hist(flow_lgths, bins=30, density=True, histtype="step",
    #                     label="All", edgecolor="r", linewidth=2)
    # plt.hist(short_flow, bins=30, density=True, histtype="step",
    #          label="Short", edgecolor="black", linewidth=2)
    # plt.hist(long_flow, bins=30, density=True, histtype="step",
    #          label="Long", edgecolor="b", linewidth=2)
    # plt.xlabel("Flow Length")

    fig, ax = plt.subplots()
    # ax.hist(exist_score, bins=25, density=True, histtype="step", cumulative=False,
    #          label="Common", edgecolor="#828282", linewidth=2)
    # ax.hist(unknw_score, bins=25, density=True, histtype="step", cumulative=False,
    #          label="Latent", edgecolor="#FE8083", linewidth=2)
    # ax.set_xlabel("Sample confidence score")
    # ax.set_ylabel("PDF")
    # plt.grid(True)

    ax.hist(tshark_score, bins=25, density=True, histtype="step", cumulative=True,
            label="tshark", edgecolor="#652d90", linewidth=2)
    ax.hist(netzob_score, bins=25, density=True, histtype="step", cumulative=True,
            label="netzob", edgecolor="#ef5a28", linewidth=2)
    ax.hist(fixed_bytes_2, bins=25, density=True, histtype="step", cumulative=True,
            label="2-bytes-fixed", edgecolor="#4dc3c3", linewidth=2)
    ax.set_xlabel("Sample confidence score")
    # plt.grid(True)
    ax.set_ylabel("CDF")

    # ----------------- cdf ------------------#
    # n, bis, p = plt.hist(tshark_score, bins=25, density=True, cumulative=True, histtype="step",
    #                      label="tshark", edgecolor="r", linewidth=2)
    # plt.hist(exist_score, bins=25, density=True, cumulative=True, histtype="step",
    #          label="Existing", edgecolor="red", linewidth=1.5)
    # plt.hist(unknw_score, bins=25, density=True, cumulative=True, histtype="step",
    #          label="Unknown", edgecolor="black", linewidth=1.5)

    # print(n)
    # print(bis)
    # print(p)

    plt.legend()
    plt.show()



