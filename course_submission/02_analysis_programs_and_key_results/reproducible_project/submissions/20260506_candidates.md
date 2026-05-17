# 20260506 候选提交文件说明

这个目录下的 `20260506_*.csv` 是本轮完整优化生成的 Kaggle 候选提交文件。

它们都来自同一次运行：

```powershell
conda run -n kaggle_house python run_optimization.py
```

所有候选文件都已经检查过：

- 行数：1459
- 列名：`Id`, `SalePrice`
- `Id` 顺序和 `sample_submission.csv` 一致
- `SalePrice` 无缺失
- `SalePrice` 无负数
- `SalePrice` 无无穷值

## 共同使用的基础模型

本轮先训练了这些模型：

| 模型 | CV RMSE | 是否进入核心融合 |
| --- | ---: | --- |
| ElasticNet | 0.10942 | 是 |
| Lasso | 0.10960 | 是 |
| SVR | 0.11020 | 是 |
| Ridge | 0.11048 | 是 |
| KernelRidge | 0.11152 | 是 |
| CatBoost | 0.11252 | 是 |
| XGBoost | 0.11291 | 是 |
| GradientBoosting | 0.11752 | 否 |
| LightGBM | 0.11841 | 否 |

核心融合池：

```text
ElasticNet, Lasso, SVR, Ridge, KernelRidge, CatBoost, XGBoost
```

## 文件逐个说明

### 1. 20260506_optimized_weight_blend.csv

推荐第一提交。

方案：非负权重优化融合。

意思是：根据 10 折 OOF 预测，自动寻找一组非负权重，让训练集 OOF RMSE 尽量低。

权重：

| 模型 | 权重 |
| --- | ---: |
| ElasticNet | 0.3508 |
| SVR | 0.3307 |
| XGBoost | 0.2593 |
| CatBoost | 0.0592 |
| Lasso | 0.0000 |
| Ridge | 0.0000 |
| KernelRidge | 0.0000 |

本地 CV：

```text
0.10609
```

这个文件最值得先提交。

### 2. 20260506_simple_blend.csv

推荐第二提交。

方案：简单平均融合。

意思是：7 个核心模型每个权重一样，直接平均它们的 log 价格预测。

本地 CV：

```text
0.10689
```

优点是非常稳，不依赖权重优化。虽然本地略低于第一候选，但 Kaggle 上可能更稳。

### 3. 20260506_inverse_cv_blend.csv

方案：按 CV 分数反比加权。

意思是：单模型 CV 越好，权重越高。

本地 CV：

```text
0.10691
```

它和 simple_blend 很接近，可作为备选。

### 4. 20260506_ridge_stack.csv

方案：Ridge stacking。

意思是：先用 7 个核心模型产生 OOF 预测，再把这些预测当作新特征，用 Ridge 做第二层融合。

本地 CV：

```text
0.10516
```

虽然它本地分数最好，但不建议第一提交，因为：

- 有负权重。
- 二层模型评分方式会偏乐观。
- Kaggle 上可能不如本地看起来那么强。

可以在前几个稳健候选提交后再试。

## clipped 版本说明

每个文件都有一个 `_clipped` 版本：

```text
20260506_optimized_weight_blend_clipped.csv
20260506_simple_blend_clipped.csv
20260506_inverse_cv_blend_clipped.csv
20260506_ridge_stack_clipped.csv
```

`clipped` 表示：对预测房价做了裁剪，上限约为：

```text
573156.41
```

用途：

- 测试是否需要压住极端高价预测。
- 如果未裁剪版本 Kaggle 分数不理想，可以提交 clipped 版本对比。

风险：

- 如果测试集里真有高价房，裁剪会低估。

## 建议提交顺序

建议先提交这四个：

1. `20260506_optimized_weight_blend.csv`
2. `20260506_simple_blend.csv`
3. `20260506_optimized_weight_blend_clipped.csv`
4. `20260506_ridge_stack.csv`

提交后请记录：

- 文件名
- Kaggle Public Score
- 排名变化

记录位置：

```text
experiments/experiment_log.csv
```
