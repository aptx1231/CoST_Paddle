# CoST_Paddle

【论文复现赛】[CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting](https://openreview.net/forum?id=PilZY3omXV2)

队伍名：aptx1231

## 方法介绍

见文件[CoST](./CoST.md)

## 数据集

参考原作者的[仓库](https://github.com/salesforce/CoST#data)，下载数据后放在`dataset/`目录即可。

## 环境配置

主要依赖为PaddlePaddle=2.3.0，Python=3.7.13。

依赖环境已放在`requirements.txt`中。

## 训练与测试

### 多变量

```shell
# ETTh1
python -u train.py ETTh1 forecast_multivar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed 0 --gpu 0 --eval > etth1_m.log
# ETTh2
python -u train.py ETTh2 forecast_multivar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed 0 --gpu 1 --eval > etth2_m.log
# ETTm1
python -u train.py ETTm1 forecast_multivar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv --repr-dims 320 --max-threads 8 --seed 0 --gpu 3 --eval > ettm1_m.log
```

### 单变量

```shell
# ETTh1
python -u train.py ETTh1 forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 0 --gpu 0 --eval > etth1_s.log
# ETTh2
python -u train.py ETTh2 forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 0 --gpu 1 --eval > etth2_s.log
# ETTm1
python -u train.py ETTm1 forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 0 --gpu 3 --eval > ettm1_s.log
```

## 参考实现

本代码主要参考以下两个仓库：

- [ts2vec](https://github.com/yuezhihan/ts2vec)
- [CoST](https://github.com/salesforce/CoST)