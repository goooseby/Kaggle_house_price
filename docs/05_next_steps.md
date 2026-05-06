# 05. Next Steps

当前 baseline 已经跑通，下一步可以围绕“提升分数”和“增强可解释性”继续做。

## 第一优先级：数值特征偏态处理

很多面积、价格相关字段是右偏的，例如：

- `LotArea`
- `GrLivArea`
- `TotalBsmtSF`
- `1stFlrSF`
- `GarageArea`
- `TotalSF`

可以对偏度较高的数值特征做：

```python
np.log1p(x)
```

这通常会提升线性模型表现。

## 第二优先级：系统调参

当前参数是经验值，不是系统搜索得到的。

建议先调：

- Ridge 的 `alpha`
- Lasso 的 `alpha`
- ElasticNet 的 `alpha` 和 `l1_ratio`
- GradientBoostingRegressor 的 `n_estimators`、`learning_rate`、`max_depth`

可以使用：

- `GridSearchCV`
- `RandomizedSearchCV`
- 手写小范围循环

由于数据量不大，调参成本可控。

## 第三优先级：加入更强模型

如果环境允许，建议加入：

- XGBoost
- LightGBM
        
这两个模型在 Kaggle 表格题上很常用。

不过需要注意：

- 它们是额外依赖
- 安装可能受本地环境影响
- 调参不当时未必超过当前 Lasso baseline

## 第四优先级：模型融合升级

当前是简单平均：

```text
final = mean(ridge, lasso, elastic_net, gbr)
```

后续可以尝试：

- 加权平均
- 根据 CV 分数设置权重
- 使用 OOF 预测训练二层模型
- StackingRegressor

但融合要小心，不要只优化本地 CV 而过拟合。

## 第五优先级：更细的特征工程

可以继续构造：

- 房屋总质量组合：`OverallQual * TotalSF`
- 年份质量组合：`OverallQual * YearBuilt`
- 是否新房
- 是否高端街区
- 地下室完成比例
- 车库面积 / 车库容量
- 门廊面积是否为 0

也可以对一些序数类别做更准确的映射，例如：

- `LotShape`
- `LandSlope`
- `BsmtExposure`
- `Functional`
- `GarageFinish`
- `PavedDrive`

## 第六优先级：可解释性报告

为了更清楚地理解模型，可以输出：

- Lasso 非零系数
- Ridge 绝对系数排序
- permutation importance
- 预测误差最大的训练样本
- 不同街区的误差分布

这部分能帮助我们判断模型学到的规律是否符合常识。

## 推荐迭代路线

建议按这个顺序走：

1. 保留当前 baseline 作为 v1
2. 增加偏态数值特征 log 变换，得到 v2
3. 调 Ridge / Lasso / ElasticNet，得到 v3
4. 尝试 XGBoost / LightGBM，得到 v4
5. 做加权融合或 stacking，得到 v5
6. 提交 Kaggle，对比 Public Score

每一步都记录 CV 分数和提交分数，避免“感觉上变复杂了，但实际没变好”。
