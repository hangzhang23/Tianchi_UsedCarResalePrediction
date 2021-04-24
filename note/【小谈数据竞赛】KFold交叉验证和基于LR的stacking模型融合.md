# 【小谈数据竞赛】KFold交叉验证和基于LR的stacking模型融合

一般在数据竞赛中用了基本模型做baseline，以及调参和特征工程做完之后，下一步可以尝试的方法一般都是融合形式，而融合也有两种，一种是同一个模型的交叉验证，另一种是不同模型的融合，一般是以stacking的形式来实现的。

## 1. KFold交叉验证

在机器学习建模过程中，通行的做法通常是将数据分为训练集和测试集。测试集是与训练独立的数据，完全不参与训练，用于最终模型的评估。在训练过程中，经常会出现过拟合的问题，就是模型可以很好的匹配训练数据，却不能很好在预测训练集外的数据。如果此时就使用测试数据来调整模型参数，就相当于在训练时已知部分测试数据的信息，会影响最终评估结果的准确性。通常的做法是在训练数据再中分出一部分做为验证(Validation)数据，用来评估模型的训练效果。

验证数据取自训练数据，但不参与训练，这样可以相对客观的评估模型对于训练集之外数据的匹配程度。模型在验证数据中的评估常用的是交叉验证，又称循环验证。它将原始数据分成K组(K-Fold)，将每个子集数据分别做一次验证集，其余的K-1组子集数据作为训练集，这样会得到K个模型。这K个模型分别在验证集中评估结果，最后的误差MSE(Mean Squared Error)加和平均就得到交叉验证误差。交叉验证有效利用了有限的数据，并且评估结果能够尽可能接近模型在测试集上的表现，可以做为模型优化的指标使用。

**交叉验证的步骤：**

1. 首先随机地将数据集切分为 k 个互不相交的大小相同的子集；

2. 然后将 k-1 个子集当成训练集训练模型，剩下的 (held out) 一个子集当测试集测试模型；

3. 将上一步对可能的 k 种选择重复进行 (每次挑一个不同的子集做测试集)；

4. 在每个训练集上训练后得到一个模型,用这个模型在相应的测试集上测试，计算并保存模型的评估指标，

5. 这样就训练了 k 个模型，每个模型都在相应的测试集上计算测试误差，得到了 k 个测试误差。

对这 k 次的测试误差取平均便得到一个交叉验证误差,并作为当前 k 折交叉验证下模型的性能指标。

在本周对二手车预测的建模过程中，除了进一步增加了更多的特征之外，还尝试了使用KFold交叉验证的方法使用xgboost建立多个模型，然后用平均的方法得到一个更为稳定的结果，从最后的结果来看还是有比较显著的提升的。

参考代码：

```python
def ensemble_model(clf, train_x, train_y, test):
    # 做五折交叉验证
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    result= []
    mean_mae = 0
    for k, (train_index, val_index) in enumerate(sk.split(train_x, train_y)):
        train_x_real = train_x.iloc[train_index]
        train_y_real = train_y.iloc[train_index]
        val_x = train_x.iloc[val_index]
        val_y = train_y.iloc[val_index]
        
        clf = clf.fit(train_x_real, train_y_real)
        val_y_pred = clf.predict(val_x)
        
        mae_val = mean_absolute_error(val_y, val_y_pred)
        print('第{}个子模型MAE:{}'.format(k+1, mae_val))
        mean_mae += mae_val / 5
        # 子模型预测测试集
        test_y_pred = clf.predict(test)
        result.append(test_y_pred)
    print(mean_mae)
    mean_result = sum(result) / 5
    return mean_result
```





## 2. 使用LR将不同模型融合

本身stacking的模型方法是可以用一个自己设定的比例来将最后的模型结果按比例融合得到新结果的。但是这个比例设定是一个比较随机的方法。而得到两个结果的权重比例又相当于对不同的x计算其系数的过程，从而可以想到使用Linear Regression来自动拟合不同模型的系数。

LR做stacking的步骤如下：

1. 将$model1, model2,...$根据训练数据得到的预测结果：$y\_1_{pred},y\_2_{pred},...$按照列组成一个trainset的x；
2. 将真实的结果$y_{true}$作为真实值组成trainset的y；
3. 用上述的trainset训练一个LR，并用训练好的LR对预测数据进行预测，得到最后的融合结果。

一般情况下stacking的结果只会有轻微的提升，但是stacking也不是所有情况下都是有效的，有时会得到一个处于各个模型结果中间的结果。

参考代码：

```python
# 对model_lgb_1预测训练集结果
x_data_stacking = pd.DataFrame(np.vstack((pred_X_data_1, pred_X_data_2)).T, columns=['lgb_1','lgb_2'])
y_data_stacking = data_y.copy(deep=True)

# 训练LR来拟合
stacking_model_lr = LinearRegression()
stacking_model_lr.fit(x_data_stacking, y_data_stacking)

# 预测
pred_y_stacking = stacking_model_lr.predict(x_test_stacking)
```

