
# SemanticAnalysis
For public sec24 meeting codes. The overall framework of our approach is shown in Figure 1.
<p align="center">
<img src=".\pic\framework.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> Field pattern-series construction process.
</p>

# Usage
- **(1) Parsing messages with Honeyeye.**
Refer to the article
```
Chuan Sheng, Yu Yao, Dongbiao Li, Hongna An, and Wei Yang. Honeyeye: A network traffic collection framework for distributed ics honeynets. In 2020 IEEE Intl
Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, SustainableComputing & Communications, Social Computing & 
Networking (ISPA/BDCloud/SocialCom/SustainCom), pages 466–473. IEEE, 2020.
```

- **(2)Constructing Pattern Series via Build_PatternSeries.**
Take the EtherNet/IP protocol as an example. The specific process is shown in the figure below.
<p align="center">
<img src=".\pic\field_series_build.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Field pattern-series construction process.
</p>

- **(3)Train the model and test the predictions.**
1. After constructing the dataset, it needs to be stored as a time series-like `.ts` file [[ref]](https://www.sktime.net/en/latest/api_reference/file_specifications/ts.html#overview) and the dataset is partitioned.
2. The model run script is:
	- TRAIN:
	```python
    --output_dir ./experiments --comment "classification for flowformer" --name ModbusTcp_fromScratch --records_file Classification_records.xls --data_dir ./Multivariate_ts/ModbusTcp --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam --pos_encoding learnable --task classification --key_metric accuracy.
	```

	- TEST:
	```python
    --output_dir ./experiments --comment "classification for flowformer" --name ModbusTcp_fromScratch --records_file Classification_records.xls --data_dir ./Multivariate_ts/ModbusTcp --data_class tsra --pattern TRAIN --val_pattern TEST --epochs 100 --lr 0.001 --batch_size 16 --optimizer RAdam --pos_encoding learnable --task classification --key_metric accuracy --test_only testset.
	```

## Contact

If you have any questions or want to use the code, please contact 2210733@stu.neu.edu.cn.
