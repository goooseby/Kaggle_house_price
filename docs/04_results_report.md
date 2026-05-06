# 04. Results Report

## 当前运行结果

最后一次运行：

```text
Baseline finished.
Train rows after outlier removal: 1458
Feature matrix: 1458 train rows x 303 columns
Best single model: lasso (0.11278 CV RMSE)
Blend CV RMSE: 0.11122
```

## 交叉验证结果

当前使用 5 折交叉验证，指标是 log 价格上的 RMSE。

| 模型 | CV RMSE |
| --- | ---: |
| Lasso | 0.11278 |
| ElasticNet | 0.11286 |
| Ridge | 0.11453 |
| GradientBoostingRegressor | 0.12177 |
| RandomForestRegressor | 0.13852 |
| Blend | 0.11122 |

## 如何理解这个分数

Kaggle 该题的评价指标等价于 log 价格上的 RMSE。

当前 blend 的本地 CV 为：

```text
0.11122
```

这已经是一个不错的第一版 baseline，说明：

- 数据清洗方向基本正确
- log 目标变换有效
- 线性正则模型很适合这个数据集
- 简单融合比单模型略好

不过本地 CV 不等于 Kaggle Public Leaderboard 分数。提交后分数可能略有差异。

## 输出文件

当前输出文件在：

```text
outputs/
```

包括：

| 文件 | 作用 |
| --- | --- |
| `eda_report.md` | 自动生成的探索性分析报告 |
| `cv_results.csv` | 每个模型的交叉验证分数 |
| `submission_baseline.csv` | Kaggle 提交文件 |

## 提交文件格式

`submission_baseline.csv` 已检查：

- 行数：1459
- 列数：2
- 列名：`Id`, `SalePrice`
- `SalePrice` 无缺失值

前几行大致类似：

```text
Id,SalePrice
1461,120842.78
1462,157291.95
1463,176832.84
1464,199964.00
```

## 当前结论

目前最强单模型是 Lasso，说明：

- one-hot 后的稀疏线性关系很强
- 正则化可以有效控制过拟合
- 对这个比赛，线性模型不是“弱模型”，反而是非常强的 baseline

GradientBoostingRegressor 有一定贡献，但单独表现不如 Lasso。

RandomForest 当前表现明显落后，原因可能是：

- 特征维度较高且 one-hot 稀疏
- 默认树模型不如正则线性模型适应该问题
- 需要更细调参才可能提升

## 风险和限制

当前 baseline 还有这些限制：

- 没有对偏态数值特征做 `log1p` 变换
- 没有系统调参
- 没有引入 XGBoost / LightGBM
- 没有做 stacking
- 部分序数类别只做了基础映射
- 没有产出图像版 EDA

这些不是错误，而是下一步优化空间。
