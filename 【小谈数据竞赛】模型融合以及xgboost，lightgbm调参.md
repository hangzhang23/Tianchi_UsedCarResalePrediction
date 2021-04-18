# 【小谈数据竞赛】模型融合以及xgboost，lightgbm调参

今年也参加了几次数据竞赛，想开一篇文章对模型融合，竞赛神器xgboost以及lightgbm的调参做一些简单梳理，总结下方法和目前得到的经验。

## 1. 模型融合

之前的两次比赛都集中在特征工程和调参上，而对于模型融合这个在数据竞赛中喜闻乐见的tricks并没有详细使用。之前也就是按照自定义的权重进行结果stacking，然而效果并不是很好==。这次参与Datawhale的二手车交易价格预测上，尝试了更多的模型融合的方法，也正好针对模型融合这个trick进行一下方法的梳理。

目前常用的模型融合方法有voting，averaging，bagging，boosting，stacking。

### 1.1 Voting

voting的思想很简单，就是谁的结果多就用谁作为最后结果，主要在分类任务中使用。例如一个二分类任务，有三个训练好的模型，针对每个测试集样本得到的结果序列$s_1{}$和$s_2$和$s_3$，对每一个结果进行投票，取结果多的为最终voting方法的结果。

### 1.2 Averaging

Averaging方法的思想也很简单，就是对多个模型的结果平均，这个主要用在回归任务中。例如得到三个训练好的回归模型，他们的结果序列是$s_1{}$和$s_2$和$s_3$，则最终的averaging结果是$\bar{s}=\frac{s_1+s_2+s_3}{3}$。

### 1.3 Bagging

Bagging就是采用有放回的方式进行抽样，用抽样的样本建立子模型,对子模型进行训练，这个过程重复多次，最后进行融合。大概分为这样两步：

1. 重复K次
   - 有放回地重复抽样建模
   - 训练子模型
2. 模型融合
   - 分类问题：voting
   - 回归问题：average

一般bagging方法也不需要自己实现，集成学习中的RandomForest就是这个方法的代表。

### 1.4 Boosting

Bagging算法可以并行处理，而Boosting的思想是一种迭代的方法，每一次训练的时候都更加关心分类错误的样例，给这些分类错误的样例增加更大的权重，下一次迭代的目标就是能够更容易辨别出上一轮分类错误的样例。最终将这些弱分类器进行加权相加。

![](https://gitee.com/zhanghang23/picture_bed/raw/master/data%20science/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/boosting_1.jpg)

![](https://gitee.com/zhanghang23/picture_bed/raw/master/data%20science/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/boosting_2.jpg)

以上两张图可以比较清晰的描述boosting的方法，而boosting方法比较有代表性的就是adaboost，GBDT，以及以gbdt为基础的xgboost，lightgbm。

### 1. 5 Stacking

Stacking方法是一种在数据竞赛中使用较多的对于结果融合的方法，其思想是用一个机器学习模型来将个体机器学习模型的结果结合爱在一起。原理图大致为这样

![](https://gitee.com/zhanghang23/picture_bed/raw/master/data%20science/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/stacking.png)

在level0处我们使用训练集训练了三个classifier，其对应输出三个output value，然后这些output value在进行集成输出给次级classifier，通过次级classifier对output数据的学习，输出的次级output则为stacking的融合结果。

而具体在stacking的操作中，我们可以参见下图

![](https://gitee.com/zhanghang23/picture_bed/raw/master/data%20science/%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0/stacking_1.jpg)

首先我们将训练集使用kfold切分为k分，每一分包括一个验证集和测试集，每次取其中k-1分训练，另外的1分用来验证，stacking是这样做的，比如对于集成的第一个模型，clf1，我们使用kfold交叉验证，那么可以得到k个clf1模型，模型的类型是一样的，但是模型里面学到的参数不一样，因为他们的训练集是不一样的，对与每一折的训练，我们还有一个验证集，那么我们用训练得到的模型在验证集合上做一次预测。因为这个时候我们的验证集还只有1f，也就是只有 $train\_set\_number/k$个样本(train_set_number表示训练样本的个数)，但总公有k折，每一折我们都会在验证集上预测，所以最终对于clf1在验证集上得到是不是train_set_number个结果。因为是kfold，所以也不用担心是没有重复的。

因为每一折的验证集样本都不会相同，也就是没有哪个样本同时出现在两个验证集上，这样下来，我们就得到第一级的结果，也是train_set_number个结果。然后在每一折上，我们在测试集上做一次预测，那么k个clf1模型预测k次得到了k个结果，也就是每一个样本预测结果有k个，我们就取一下平均，看到是取平均，这样取完平均以后每一个样本在clf1模型上就得到一个预测结果。这只是一个模型的过程，因为我们需要集成很多个模型，那么我重复n个模型，做法和上面是一样的，假设我们有n个模型，我们stacking第一层出来，在验证集上得到的结果特征的维度是(train_set_number, n)，这就是我们对第一层结果的一个特征堆叠方法，这样第一层出来的结果又可以作为特征训练第二层，第二层仍然可以stacking多个模型，或者直接接一个模型用于训练，然后直接预测。那么同样，对于测试集第一层出来的维度(test_set_number, n)，也就是测试集样本的行数，可以用第二层训练的模型在这个上面预测，得到我们最后的结果。这个就是stacking的整个过程。

## 2. xgboost和lightgbm调参步骤

这两个集成学习模型是数据竞赛中使用非常频繁的两个模型，他们都是从gbdt衍生出来的梯度提升树模型，但是又针对gbdt的一些不能并行化和速度慢的缺点进行了针对性的增强，使得他们在各种数据竞赛中大放异彩。这里把我自己比较常用的xgboost和lightgbm调参步骤总结下来，一般会使用GridSearchCV来查询范围内最佳参数。

### 2.1 xgboost调参步骤

1. 先确定eta（学习率），0.01到0.3之间选取合适较大值。
2. 调整max_depth，控制树深度，初始10，也可以在5-10之间网格搜索最佳。
3. 调整subsample，控制样本抽样率，在0.3-1之间网格搜索最佳。
4. 调整min_child_weight，越大模型越保守。
5. 调整colsample_bytree,特征子抽样，在0.3-1之间网格搜索。
6. 再次降低eta求精确模型。

### 2.2 lightgbm调参步骤

1. 先确定learning_rate，一般都在0.01到0.3之间选，可刚开始选较大便于快速收敛；n_estimators为迭代数量，这个根据经验调整，可用早停法截取。
2. 调整max_depth和max_leaves，这两个控制树深度和模型复杂度，一般后者小于$2^{max\_depth}$.
3. 调整min_child_sample和min_child_weight,这两个也是控制过拟合用的。
4. 调整colsample_bytree和subsample，特征子抽样和样本采样，也是空值过拟合用。
5. 调整reg_alpha和reg_lambda，分别是L1和L2正则，控制过拟合。
6. 再次降低learning_rate求的最精确模型。