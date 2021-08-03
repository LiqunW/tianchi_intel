#代码使用说明
##数据准备

1.create_gfile.py 生成标签

2.create_lmdb_dataset.py 转lmdb格式


##训练
1.修改train_config文件

2.train.py文件训练


##########分割线##########

#华为ocr菜单识别比赛
2021年7月31日

数据集情况
1.存在超长文本，如label长度最长为300

2.存在曲面文本，艺术字文本，多国语言（如西班牙、法国）

3.label为###的图片需要过滤