# Tianchi_UsedCarResalePrediction

这个repo是来自[天池比赛-二手车交易价格预测](https://tianchi.aliyun.com/competition/entrance/231784/introduction)。赛题以二手车市场为背景，要求选手预测二手汽车的交易价格，这是一个典型的回归问题。


这次竞赛集中做的任务：
1. 数据探索：查看缺失值，查看数据分布；
2. 属于预处理：数据格式转换，填补缺失值；
3. 建立模型。
    - xgboost；
    - lightgbm；
    - 收集上面两个模型的训练集结果并汇总，使用lr进行stacking融合；
    - 构建神经网络。

其中大多数结果集中在500-600之间，很遗憾忙于工作对于调参和特征工程部份没有充足时间研究，之后时间充裕后会进一步对结果进行优化。


----
**4月25日**
添加了kflod的交叉验证的xgb方案，并且加入了brand相关特征以及时间相关特征。
