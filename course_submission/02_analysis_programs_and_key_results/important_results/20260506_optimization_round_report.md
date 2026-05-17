# 20260506 完整优化回合报告

## 本轮做了什么

本轮不是只改一个模型，而是做了一次完整优化：

1. 使用增强版预处理生成特征矩阵。
2. 对偏态明显的数值特征做 `log1p` 变换。
3. 补充更完整的序数类别编码。
4. 增加缺失指示、面积组合、比例特征、质量交互特征。
5. 使用 10 折 OOF 验证训练多个模型。
6. 保存每个模型的 OOF 预测和测试集预测。
7. 比较多种融合方案。
8. 生成多个 Kaggle 候选提交文件。

本轮入口：

```powershell
conda run -n kaggle_house python run_optimization.py
```

核心代码：

- `house_price/advanced_preprocessing.py`
- `house_price/advanced_modeling.py`
- `run_optimization.py`

## 和 baseline 的对比

| 项目 | 本地 CV RMSE | Kaggle Public Score |
| --- | ---: | ---: |
| 第一版 baseline blend | 0.11122 | 0.12859 |
| 本轮推荐融合 optimized_weight_blend | 0.10609 | 待提交 |

本地 CV 看，本轮比 baseline 有明显提升。不过 Kaggle 分数还需要提交后验证。

## 本轮特征工程

增强版特征矩阵规模：

| 数据 | 行数 | 特征数 |
| --- | ---: | ---: |
| train | 1458 | 447 |
| test | 1459 | 447 |

相比 baseline 的 303 个特征，本轮增加到了 447 个特征。

主要新增内容：

- 缺失值指示特征，例如 `LotFrontage_WasMissing`
- 总面积特征，例如 `TotalSF`
- 总居住相关面积，例如 `TotalLivingSF`
- 总完成面积，例如 `TotalFinishedSF`
- 总卫浴数，例如 `TotalBathrooms`
- 门廊总面积，例如 `TotalPorchSF`
- 房龄和翻新年龄，例如 `HouseAge`、`RemodAge`
- 比例特征，例如 `FinishedBasementRatio`、`GarageAreaPerCar`
- 质量交互特征，例如 `OverallQual_TotalSF`
- 街区和质量组合特征，例如 `Neighborhood_OverallQual`

做了 `log1p` 变换的数值特征共 27 个：

```text
LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
1stFlrSF, 2ndFlrSF, LowQualFinSF, GrLivArea, WoodDeckSF, OpenPorchSF,
EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, TotalSF,
TotalLivingSF, TotalFinishedSF, TotalPorchSF, PorchRatio,
LotAreaPerLivingSF, OverallQual_TotalSF, OverallQual_GrLivArea,
OverallQual_TotalBathrooms, Age_OverallQual
```

## 单模型结果

所有模型都使用同一套增强特征，目标值为 `log1p(SalePrice)`。

| 模型 | 本地 10 折 CV RMSE | 说明 |
| --- | ---: | --- |
| ElasticNet | 0.10942 | 本轮最强单模型，正则线性模型 |
| Lasso | 0.10960 | 稀疏线性模型，表现稳定 |
| SVR | 0.11020 | 支持向量回归，小数据上常有竞争力 |
| Ridge | 0.11048 | 稳定线性模型 |
| KernelRidge | 0.11152 | 核岭回归，捕捉非线性 |
| CatBoost | 0.11252 | 梯度提升树模型 |
| XGBoost | 0.11291 | 梯度提升树模型 |
| GradientBoosting | 0.11752 | sklearn 提升树 |
| LightGBM | 0.11841 | 本轮参数下表现一般 |

结论：

- 这道题上正则线性模型仍然非常强。
- 强树模型并没有单独超过 ElasticNet/Lasso，但它们能提供不同视角，适合用于融合。
- LightGBM 和 sklearn GBR 本轮单模型较弱，所以没有放进最终推荐融合的核心权重。

## 融合方案结果

进入融合池的模型：

```text
elastic_net, lasso, svr, ridge, kernel_ridge, catboost, xgboost
```

没有进入核心融合池的模型：

```text
gbr, lightgbm
```

原因：这两个模型本地 CV 明显弱于前面的模型，加入后可能增加噪声。

| 融合方案 | 本地 CV RMSE | 说明 |
| --- | ---: | --- |
| ridge_stack | 0.10516 | 二层 Ridge stacking，分数最低但偏乐观 |
| optimized_weight_blend | 0.10609 | 非负权重优化融合，推荐第一提交 |
| simple_blend | 0.10689 | 简单平均，稳健备选 |
| inverse_cv_blend | 0.10691 | 按 CV 分数反比加权 |

## 各融合方案是什么意思

### optimized_weight_blend

文件：

```text
submissions/20260506_optimized_weight_blend.csv
```

这是本轮推荐第一提交。

它使用 OOF 预测搜索非负权重，权重如下：

| 模型 | 权重 |
| --- | ---: |
| ElasticNet | 0.3508 |
| SVR | 0.3307 |
| XGBoost | 0.2593 |
| CatBoost | 0.0592 |
| Lasso | 0.0000 |
| Ridge | 0.0000 |
| KernelRidge | 0.0000 |

这个方案的特点：

- 权重非负，比较稳。
- 自动选择了互补性较强的模型。
- 本地 CV 明显优于单模型。
- 不像 stacking 那样有负权重和额外偏乐观风险。

### simple_blend

文件：

```text
submissions/20260506_simple_blend.csv
```

这是 7 个核心模型的简单平均：

```text
ElasticNet + Lasso + SVR + Ridge + KernelRidge + CatBoost + XGBoost
```

每个模型权重都是 `1/7`。

这个方案的特点：

- 逻辑最简单。
- 不依赖权重搜索。
- 本地略弱于 optimized_weight_blend。
- Kaggle 上有时反而更稳。

### inverse_cv_blend

文件：

```text
submissions/20260506_inverse_cv_blend.csv
```

这是根据每个模型的 CV 分数自动分配权重。CV 分数越低，权重越高。

这个方案的特点：

- 比 simple_blend 多考虑了单模型强弱。
- 但它只根据单模型分数分配权重，没有直接优化组合误差。
- 本轮本地分数和 simple_blend 很接近。

### ridge_stack

文件：

```text
submissions/20260506_ridge_stack.csv
```

这是二层 Ridge stacking：

1. 第一层模型先产生 OOF 预测。
2. 把这些 OOF 预测当作新特征。
3. 用 Ridge 学习第二层融合模型。
4. 再对测试集做融合预测。

它的本地 CV 最低，但需要谨慎：

- 它出现了负权重。
- 它是在 OOF 预测矩阵上训练二层模型，又在同一个矩阵上评分。
- 所以这个 `0.10516` 很可能偏乐观。

它可以提交测试，但不建议作为第一优先级。

## clipped 文件是什么意思

每个融合方案都有一个 `_clipped` 版本，例如：

```text
20260506_optimized_weight_blend_clipped.csv
```

`clipped` 表示对最终预测价格做了上限/下限裁剪。

本轮裁剪上限约为训练集房价的 `99.7%` 分位数：

```text
573156.41
```

为什么要做裁剪：

- 第一版 baseline 曾经预测出约 `1,347,535` 的极端高价。
- 这可能导致 Kaggle 分数变差。
- 裁剪版本用于测试“压住极端高价”是否有帮助。

为什么不一定应该优先提交 clipped：

- 测试集中确实可能存在高价房。
- 裁剪可能把真实豪宅压低。
- 所以 clipped 是验证用候选，不是默认最优。

## 候选提交文件说明

| 文件 | 方案 | 是否裁剪 | 推荐程度 |
| --- | --- | --- | --- |
| `20260506_optimized_weight_blend.csv` | 非负权重优化融合 | 否 | 第一优先级 |
| `20260506_simple_blend.csv` | 7 模型简单平均 | 否 | 第二优先级 |
| `20260506_optimized_weight_blend_clipped.csv` | 非负权重优化融合 | 是 | 第三优先级 |
| `20260506_ridge_stack.csv` | Ridge stacking | 否 | 第四优先级 |
| `20260506_inverse_cv_blend.csv` | 按 CV 反比加权 | 否 | 可选 |
| `20260506_simple_blend_clipped.csv` | 简单平均 | 是 | 可选 |
| `20260506_inverse_cv_blend_clipped.csv` | CV 反比加权 | 是 | 可选 |
| `20260506_ridge_stack_clipped.csv` | Ridge stacking | 是 | 可选 |

## 推荐提交顺序

建议按这个顺序提交：

1. `submissions/20260506_optimized_weight_blend.csv`
2. `submissions/20260506_simple_blend.csv`
3. `submissions/20260506_optimized_weight_blend_clipped.csv`
4. `submissions/20260506_ridge_stack.csv`

提交后把 Public Score 记录回：

```text
experiments/experiment_log.csv
```

## 输出文件位置

自动结果：

```text
outputs/optimized_cv_results.csv
outputs/optimized_blend_results.csv
outputs/optimized_prediction_diagnostics.csv
```

候选提交：

```text
submissions/20260506_*.csv
```

人工报告：

```text
reports/20260506_optimization_round_report.md
submissions/20260506_candidates.md
```

## Kaggle 反馈

已提交部分候选文件，Public Score 如下：

| 文件 | Public Score |
| --- | ---: |
| `20260506_optimized_weight_blend_clipped.csv` | 0.11865 |
| `20260506_ridge_stack_clipped.csv` | 0.11982 |
| `20260506_simple_blend.csv` | 0.12153 |
| `20260506_optimized_weight_blend.csv` | 0.12173 |
| `20260506_ridge_stack.csv` | 0.12258 |
| 第一版 baseline | 0.12859 |

## Kaggle 反馈复盘

### 结论 1：本轮优化是有效的

第一版 baseline 的 Public Score 是：

```text
0.12859
```

本轮最好成绩是：

```text
0.11865
```

提升约：

```text
0.00994
```

这说明增强特征、更多模型和融合方向整体是有效的。

### 结论 2：裁剪高价预测非常有效

同一模型方案对比：

| 方案 | 未裁剪 | 裁剪后 | 改善 |
| --- | ---: | ---: | ---: |
| optimized_weight_blend | 0.12173 | 0.11865 | 0.00308 |
| ridge_stack | 0.12258 | 0.11982 | 0.00276 |

这说明测试集 Public 部分对极端高价预测比较敏感。本轮裁剪不是小修饰，而是明确有效。

下一步应重点研究：

- 哪些测试样本被裁剪影响最大
- 裁剪上限是否可以调优
- 是否应该用更温和的高价校准，而不是硬裁剪

### 结论 3：ridge_stack 本地 CV 明显偏乐观

`ridge_stack` 的本地 CV 是：

```text
0.10516
```

但 Public Score 是：

```text
0.12258
```

它是本地最强、线上最弱的已提交候选。这个结果验证了之前的判断：当前 stacking 评分方式偏乐观，不应该作为主线。

后续如果继续做 stacking，需要使用更严格的嵌套 OOF 或独立验证方式。

### 结论 4：simple_blend 比 optimized_weight_blend 未裁剪版本更稳一点

未裁剪情况下：

| 文件 | Public Score |
| --- | ---: |
| simple_blend | 0.12153 |
| optimized_weight_blend | 0.12173 |

虽然本地 CV 里 optimized_weight_blend 更好，但 Public Score 略差。这说明 OOF 权重搜索可能有轻微过拟合，简单平均更稳。

不过裁剪之后，optimized_weight_blend 是目前最好结果。

### 结论 5：本地 CV 仍然整体偏乐观

本地 CV 与 Public Score 差距：

| 方案 | 本地 CV | Public Score | 差距 |
| --- | ---: | ---: | ---: |
| optimized_weight_blend | 0.10609 | 0.12173 | 0.01564 |
| optimized_weight_blend_clipped | 0.10609 | 0.11865 | 0.01256 |
| simple_blend | 0.10689 | 0.12153 | 0.01464 |
| ridge_stack | 0.10516 | 0.12258 | 0.01742 |

后续优化不能只看本地 CV，需要把高价预测分布、裁剪策略和 Public Score 反馈一起纳入判断。

## 下一步建议

下一轮不要先盲目加模型，而应该围绕“裁剪为什么有效”展开：

1. 分析所有候选在高价测试样本上的预测差异。
2. 尝试多个裁剪上限，例如训练集 99.0%、99.3%、99.5%、99.7%、99.9% 分位数。
3. 生成 `simple_blend_clipped` 和 `inverse_cv_blend_clipped` 的提交结果，验证“裁剪收益是否普遍存在”。
4. 改造 stacking 验证方式，避免继续被偏乐观 CV 误导。
5. 让融合权重更保守，降低 OOF 权重搜索过拟合。
