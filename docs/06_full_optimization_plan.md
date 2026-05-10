# 06. Full Optimization Plan

## 背景

第一版 baseline 已经提交到 Kaggle：

| 指标 | 数值 |
| --- | ---: |
| Local CV RMSE | 0.11122 |
| Kaggle Public Score | 0.12859 |
| Public Rank | 约 30% |

这个结果说明 baseline 方向是对的，但本地 CV 明显比 Kaggle Public Score 乐观。下一步不能只是堆复杂度，而要在同一个优化回合里同时处理验证、特征、模型、调参和融合。

## 工作原则

本项目后续不按“为了分阶段而分阶段”的方式推进。

真正的边界是：

- 当前能做的，就在本轮尽量做完。
- 只有当继续推进依赖 Kaggle 分数、环境安装结果或新的实验反馈时，才停下来。
- 每次提交前必须能回答：这份提交和上一份相比到底改了什么，为什么可能更好。

因此，下一轮不是简单的 “v2 特征版”，而是一个完整优化回合。

## 本轮目标

本轮目标是产出一组高质量候选提交，而不是只产出一个文件。

希望达到：

- 训练过程记录更清楚
- 单模型表现更强
- 融合方式更稳
- 候选提交数量控制在 2 到 4 个
- 每个候选都有清楚说明

Public Score 目标：

- 保守目标：优于 `0.12859`
- 合理目标：接近或低于 `0.120`
- 进阶目标：接近 `0.115`

## 本轮要做的事

### 1. 重建实验与验证体系

必须做：

- 统一随机种子
- 固定 KFold 或 RepeatedKFold 策略
- 每个模型保存 OOF 预测
- 每个模型保存测试集预测
- 记录每个实验的本地 CV
- 记录提交后的 Public Score

要额外分析：

- 本地 CV 与 Public Score 的差距
- 训练集中误差最大的样本
- 测试集中预测最高和最低的样本
- 极端预测是否需要裁剪

### 2. 改进预处理和特征工程

当前 baseline 已经有基本缺失值处理和简单特征工程，但还不够细。

本轮应一次性补充：

- 数值偏态特征 `log1p` 变换
- 更完整的序数类别映射
- 缺失指示特征
- 面积组合特征
- 年龄类特征
- 比例类特征
- 质量与面积的交互特征
- 街区相关特征

重点候选特征：

| 特征 | 意义 |
| --- | --- |
| `TotalSF` | 总面积 |
| `TotalLivingSF` | 地上居住相关面积 |
| `TotalBathrooms` | 总卫浴数 |
| `FinishedBasementRatio` | 地下室完成比例 |
| `GarageAreaPerCar` | 单车位车库面积 |
| `OverallQual_TotalSF` | 质量和面积交互 |
| `OverallQual_GrLivArea` | 质量和居住面积交互 |
| `OverallQual_GarageCars` | 质量和车库容量交互 |
| `HouseAge` | 房龄 |
| `RemodAge` | 距离翻新的时间 |
| `IsNewHouse` | 是否接近新房 |

### 3. 引入更强模型

当前模型只有 scikit-learn 基础模型。下一轮建议引入：

- KernelRidge
- SVR
- XGBoost
- LightGBM
- CatBoost

保留并调优：

- Ridge
- Lasso
- ElasticNet
- GradientBoostingRegressor

候选模型池：

| 模型 | 作用 |
| --- | --- |
| Lasso | 稀疏线性强 baseline |
| ElasticNet | 稳定线性模型 |
| Ridge | 稳定线性模型 |
| KernelRidge | 捕捉非线性，经典强模型 |
| SVR | 小数据集上可能很强 |
| GradientBoosting | sklearn 树提升基线 |
| XGBoost | 表格题强模型 |
| LightGBM | 表格题强模型 |
| CatBoost | 对类别关系友好 |

### 4. 做系统调参

数据量很小，可以使用较充分的搜索。

建议使用：

- `Optuna` 做主要模型调参
- 或者对线性模型使用小范围网格搜索

优先调：

- Lasso / ElasticNet 的正则强度
- KernelRidge 的 `alpha`、`kernel`、`degree`、`coef0`
- SVR 的 `C`、`epsilon`、`gamma`
- XGBoost / LightGBM / CatBoost 的树深、学习率、采样率、正则项

注意：

- 调参以 OOF CV 为准。
- 不为了本地 CV 极限而牺牲泛化稳定性。
- 参数结果要写入实验报告。

### 5. 融合和裁剪

本轮不只做简单平均。

需要比较：

- 单模型提交
- 简单平均
- 按 CV 反比加权平均
- 在 OOF 上搜索最优非负权重
- 二层 Ridge / ElasticNet stacking
- 对极端预测做轻度裁剪的版本

特别关注：

第一版提交中测试集最高预测约 `1,347,535`，高于训练集最高房价 `755,000`。这不一定错，但需要比较：

- 不裁剪版本
- 按训练集 99.7% 分位数轻度裁剪
- 只对极端高价样本做稳健处理

### 6. 输出候选提交和报告

本轮完成后至少输出：

- 单模型 CV 排名
- 融合方案 CV 排名
- 候选提交文件
- 实验报告
- 更新后的实验记录

候选提交数量建议控制在 2 到 4 个：

- 最佳 OOF 融合
- 最佳 OOF 融合 + 裁剪
- 最佳单模型
- 一个多样性较强的备选融合

## 本轮停止点

只有在以下位置停下来等待验证：

1. 环境安装完成前
2. 候选提交文件生成后
3. Kaggle Public Score 返回后

Kaggle 分数返回后，下一轮重点不是继续盲目加东西，而是分析：

- 哪个候选提交最好
- 本地 CV 是否仍然偏乐观
- 裁剪是否有效
- 强模型是否真的带来提升
- 下一轮应该加强特征、模型还是验证方式

## 预期交付物

下一轮完整优化回合结束时，应有：

```text
experiments/experiment_log.csv
reports/<date>_optimization_round_report.md
submissions/<date>_best_blend.csv
submissions/<date>_best_blend_clipped.csv
```

以及必要的代码改动。
