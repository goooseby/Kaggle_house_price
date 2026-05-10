# 20260507 Target Encoding 实验报告

## 1. 实验目的

本轮正式执行 Target Encoding 模型改造主题。

目标不是继续微调上一轮 q993 裁剪比例，而是验证：类别变量的目标均值信息是否能给现有高级特征体系带来结构性增量。

## 2. 本轮完整方案

### 2.1 基础特征

沿用 `house_price.advanced_preprocessing.build_advanced_feature_matrix` 生成的高级特征矩阵，包括缺失值处理、偏态数值变换、人工组合特征和独热编码等。

### 2.2 新增 Target Encoding 特征

本轮生成 TE 特征数：`20`

TE 特征包括：

TE_Neighborhood, TE_MSSubClass, TE_HouseStyle, TE_Exterior1st, TE_Exterior2nd, TE_SaleType, TE_SaleCondition, TE_Condition1, TE_Foundation, TE_GarageType, TE_RoofStyle, TE_BldgType, TE_Neighborhood__OverallQual, TE_Neighborhood__MSSubClass, TE_Neighborhood__HouseStyle, TE_OverallQual__MSSubClass, TE_OverallQual__GarageCars, TE_OverallQual__ExterQual, TE_OverallQual__KitchenQual, TE_SaleCondition__SaleType

训练集 TE 使用 OOF 方式生成，测试集 TE 使用全量训练统计并做平滑，避免目标泄漏。

编码后做安全 Robust 缩放：按中位数居中、按 IQR 缩放，但缩放分母下限固定为 0.05 个 log 点，避免低方差类别列被异常放大。

### 2.3 单模型

本轮训练 7 个单模型：`te_ridge`、`te_elastic_net`、`te_kernel_ridge`、`te_svr`、`te_xgboost`、`te_lightgbm`、`te_catboost`。

### 2.4 融合方案

融合只使用上一轮经验中更稳的模型族：ElasticNet、Ridge、SVR、XGBoost、CatBoost。

- `te_simple_blend`：五个模型等权融合。
- `te_weighted_blend`：基于 OOF RMSE 优化权重，单模型最高权重限制为 0.35。
- `te_conservative_blend`：50% 等权融合 + 50% 优化权重融合，用来降低权重优化过拟合风险。

### 2.5 提交文件变体

每个融合方案会输出三类提交文件：

- 原始版本：只使用本轮 TE 融合预测。
- `_clip_q993`：按训练集 SalePrice 的 99.3% 分位数做硬裁剪，沿用上一轮已验证更稳的高价控制策略。
- `_mix_current_best_clip_q993`：先做 q993 裁剪，再与当前公开最好文件 `20260507_opt_clip_q993.csv` 做 50/50 log 融合，用于测试 TE 是否提供增量信息。

## 3. 本地训练记录

本轮保留 OOF CV RMSE 作为训练过程记录，但最终方案优劣以 Kaggle Public Score 为准。

| candidate | kind | cv_rmse | weights |
| --- | --- | ---: | --- |
| te_weighted_blend | blend | 0.106527 | te_elastic_net:0.3051; te_ridge:0.0000; te_svr:0.3500; te_xgboost:0.2864; te_catboost:0.0586 |
| te_conservative_blend | blend | 0.106757 | te_elastic_net:0.2525; te_ridge:0.1000; te_svr:0.2750; te_xgboost:0.2432; te_catboost:0.1293 |
| te_simple_blend | blend | 0.107108 | te_elastic_net:0.2000; te_ridge:0.2000; te_svr:0.2000; te_xgboost:0.2000; te_catboost:0.2000 |
| te_svr | single_model | 0.110194 |  |
| te_elastic_net | single_model | 0.110742 |  |
| te_ridge | single_model | 0.111827 |  |
| te_catboost | single_model | 0.112855 |  |
| te_xgboost | single_model | 0.113070 |  |
| te_kernel_ridge | single_model | 0.113380 |  |
| te_lightgbm | single_model | 0.115939 |  |

## 4. 候选提交文件

| candidate | path |
| --- | --- |
| te_simple_blend | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_simple_blend.csv |
| te_simple_blend_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_simple_blend_clip_q993.csv |
| te_simple_blend_mix_current_best_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_simple_blend_mix_current_best_clip_q993.csv |
| te_weighted_blend | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_weighted_blend.csv |
| te_weighted_blend_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_weighted_blend_clip_q993.csv |
| te_weighted_blend_mix_current_best_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_weighted_blend_mix_current_best_clip_q993.csv |
| te_conservative_blend | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_conservative_blend.csv |
| te_conservative_blend_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_conservative_blend_clip_q993.csv |
| te_conservative_blend_mix_current_best_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_conservative_blend_mix_current_best_clip_q993.csv |

## 5. 推荐优先提交顺序

| candidate | path |
| --- | --- |
| te_conservative_blend_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_conservative_blend_clip_q993.csv |
| te_weighted_blend_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_weighted_blend_clip_q993.csv |
| te_conservative_blend_mix_current_best_clip_q993 | D:\pythonDev\PythonProject\Kaggle_house_price\submissions\model_redesign_20260507\target_encoding\20260507_te_conservative_blend_mix_current_best_clip_q993.csv |

推荐逻辑：

- 优先选择 `clip_q993` 版本，因为已有验证表明 q993 高价硬裁剪有效。
- 优先选择 conservative/weighted blend，而不是单模型。
- 与当前最好提交做 50/50 log 融合的文件用于测试 TE 模型是否提供增量信息。

## 6. 文件追溯

- 训练入口：`run_target_encoding_experiment.py`
- Target Encoding 特征：`house_price/target_encoding.py`
- 实验中间产物：`experiments\model_redesign_20260507\target_encoding`
- 提交候选文件：`submissions\model_redesign_20260507\target_encoding`

## 7. 初步结论

本轮 TE 方案在融合后取得了较好的本地 CV：`te_weighted_blend` 为 `0.106527`，`te_conservative_blend` 为 `0.106757`。但上一轮 optimized 融合的本地 CV 约为 `0.10609`，因此从普通 CV 看，TE 不是明确提升。

单模型也呈现类似现象：ElasticNet、Ridge、CatBoost 等加入 TE 后没有比上一轮对应模型更强；SVR 基本持平。这说明本轮 TE 的主要价值不一定是单独替代原方案，而更可能体现在“与当前最好方案做融合时是否提供不同误差信息”。

由于上一轮已经验证 q993 高价裁剪有效，所以正式提交不推荐原始版本，优先测试 q993 裁剪版本和与当前最好文件融合的版本。

推荐提交优先级：

1. `20260507_te_conservative_blend_mix_current_best_clip_q993.csv`：最适合测试 TE 是否提供增量信息，风险相对低。
2. `20260507_te_conservative_blend_clip_q993.csv`：纯 TE 保守融合裁剪版，用来验证 TE 独立方案质量。
3. `20260507_te_weighted_blend_mix_current_best_clip_q993.csv`：更偏优化权重，适合作为第三个探索文件。
4. `20260507_te_weighted_blend_clip_q993.csv`：纯 TE 优化权重裁剪版，可能略激进。

如果这些文件的 Public Score 不能接近或超过 `0.11805`，则说明当前 TE 组合不是高价值方向，下一轮应转向“原生类别模型/更暴力的模型族融合”，例如 CatBoost 原生类别特征、不同特征空间的模型族差异化训练，而不是继续微调 TE 特征列表。

## 8. Kaggle Public Score 反馈

本轮实际提交 4 个文件，结果如下：

| 文件 | Public Score | 结论 |
| --- | --- | --- |
| `20260507_te_weighted_blend_clip_q993.csv` | `0.11819` | 纯 TE 优化权重融合，略差于上一轮最好 `0.11805` |
| `20260507_te_weighted_blend_mix_current_best_clip_q993.csv` | `0.11774` | 与当前最好方案融合后明显提升 |
| `20260507_te_conservative_blend_clip_q993.csv` | `0.11802` | 纯 TE 保守融合，已经略好于上一轮 `0.11805` |
| `20260507_te_conservative_blend_mix_current_best_clip_q993.csv` | `0.11765` | 本轮最优，也是当前最好成绩 |

### 8.1 对预期的回顾

实验前我们的判断是：TE 的本地 CV 没有明显超过上一轮 optimized 融合，因此不应直接视为新主线；更关键的是看它与当前最好提交做 50/50 log 融合后是否提供增量。

Public Score 验证了这个判断。纯 TE 文件提升有限，`weighted_clip_q993` 甚至回落到 `0.11819`；但两个 mix 文件都明显优于对应纯 TE 文件，也优于上一轮最好 `0.11805`。

### 8.2 初步解释

TE 特征捕捉到了一部分类别分组的价格水平信息，但单独使用时仍然受高价尾部和模型误差结构限制。它更有价值的地方是提供了与上一轮 optimized 方案不同的误差信号，因此融合后可以降低部分样本误差。

保守融合优于优化权重融合，也说明这道题当前阶段的收益更来自稳健差异化，而不是把 OOF 权重优化到极致。

### 8.3 下一步建议

下一轮不建议继续围绕 q993 裁剪比例做微调。更值得做的是扩大模型差异性：

- CatBoost 原生类别特征方案，不再完全依赖 one-hot 后的数值矩阵。
- 构造不同特征空间的模型族，例如原始类别特征、TE 特征、高级数值组合特征分别训练。
- 融合时优先做稳健多源融合，而不是单纯追求 OOF 最低。
