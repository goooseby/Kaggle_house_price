# Kaggle 房价预测期末报告大纲

## 0. 写作目标

本报告用于总结 Kaggle House Prices 房价预测练习赛的完整数据挖掘流程。报告目标不是只展示最终分数，而是清楚说明：

- 我们面对的预测问题是什么。
- 数据有什么特点。
- 通过 EDA 得到了哪些认识。
- 如何进行数据清洗、预处理和特征工程。
- 使用了哪些模型和融合策略。
- 每轮实验带来了什么改进。
- 最终方案为什么合理，结果是否可以作为阶段最终结果。

最终报告建议控制在“结构清楚、图表充分、结论明确”的风格，不需要把所有代码细节写成流水账。

## 1. 项目背景与任务说明

### 1.1 任务来源

说明本项目来源于 Kaggle 入门竞赛 **House Prices - Advanced Regression Techniques**，任务是根据房屋属性预测房屋销售价格。

### 1.2 预测目标

预测目标变量为 `SalePrice`，即房屋销售价格。

需要说明房价预测具有典型的结构化表格数据特点：既包含数值型变量，也包含大量类别型变量，还存在缺失值、异常值和偏态分布。

### 1.3 评价指标

Kaggle 使用 RMSLE 作为评价指标。由于 RMSLE 等价于对价格取 log 后计算 RMSE，因此本项目在建模时对目标变量使用 `log1p(SalePrice)` 变换。

本章可引用：

- `main.py`
- `house_price/config.py`
- `experiments/experiment_log.csv`

## 2. 数据集说明

### 2.1 数据规模

说明训练集、测试集的样本数量和特征数量。

建议引用 EDA 中的数据概览表：

- `reports/eda/tables/dataset_overview.csv`
- `reports/eda/tables/dtype_summary.csv`

### 2.2 特征类型

说明数据包含：

- 房屋面积类数值特征，例如 `GrLivArea`、`TotalBsmtSF`
- 房屋质量类等级特征，例如 `OverallQual`、`ExterQual`、`KitchenQual`
- 地理位置类类别特征，例如 `Neighborhood`
- 年份类特征，例如 `YearBuilt`、`YearRemodAdd`
- 交易条件类特征，例如 `SaleType`、`SaleCondition`

### 2.3 缺失值情况

总结训练集和测试集的缺失值特点。重点说明部分缺失并不一定表示数据错误，而是表示“没有该设施”，例如没有车库、没有地下室、没有泳池等。

建议引用：

- `reports/eda/figures/missing_train.png`
- `reports/eda/figures/missing_test.png`
- `reports/eda/tables/missing_train.csv`
- `reports/eda/tables/missing_test.csv`

## 3. 探索性数据分析 EDA

这一章应当作为报告的图表重点。已有 EDA 图表比较完整，后续写正文时可以直接引用。

### 3.1 目标变量分布

说明 `SalePrice` 原始分布右偏明显，高价房形成长尾；取 log 后分布更接近正态，更适合回归建模。

建议引用：

- `reports/eda/figures/target_distribution.png`
- `reports/eda/figures/target_log_distribution.png`
- `reports/eda/tables/target_summary.csv`

### 3.2 重要数值特征与价格关系

重点分析：

- `OverallQual` 与房价强相关。
- `GrLivArea` 与房价整体正相关，但存在异常点。
- `TotalBsmtSF` 与房价也有明显关系。
- 年份类变量反映房屋新旧程度，对价格有影响。

建议引用：

- `reports/eda/figures/overallqual_boxplot.png`
- `reports/eda/figures/grlivarea_saleprice.png`
- `reports/eda/figures/totalbsmtsf_saleprice.png`
- `reports/eda/figures/yearbuilt_saleprice.png`
- `reports/eda/figures/yearremod_saleprice.png`
- `reports/eda/tables/numeric_correlations.csv`

### 3.3 类别特征与价格关系

重点分析：

- `Neighborhood` 不同社区的中位房价差异明显。
- `ExterQual`、`KitchenQual` 等质量类特征与房价有明显分层关系。
- 类别变量蕴含重要信息，因此后续尝试了 one-hot、有序编码和 Target Encoding。

建议引用：

- `reports/eda/figures/neighborhood_median_price.png`
- `reports/eda/figures/saleprice_by_exterqual.png`
- `reports/eda/figures/saleprice_by_house_style.png`
- `reports/eda/tables/neighborhood_price.csv`
- `reports/eda/tables/categorical_cardinality.csv`

### 3.4 训练集与测试集分布差异

说明训练集和测试集整体分布相近，但部分变量仍存在一定差异，所以需要在预处理时保证 train/test 一致变换。

建议引用：

- `reports/eda/figures/train_test_grlivarea.png`
- `reports/eda/figures/train_test_lotarea.png`
- `reports/eda/figures/train_test_overallqual.png`
- `reports/eda/tables/train_test_numeric_drift.csv`
- `reports/eda/tables/train_test_categorical_drift.csv`

### 3.5 异常值分析

说明 `GrLivArea` 特别大但 `SalePrice` 较低的样本可能影响模型拟合，因此在训练中删除了已知异常点。

建议引用：

- `reports/eda/figures/grlivarea_saleprice.png`
- `reports/eda/tables/outlier_candidates.csv`
- `house_price/data.py`

## 4. 数据预处理

这一章是后续需要重点补写的内容。

### 4.1 目标变量变换

对 `SalePrice` 使用 `log1p`，预测后再用 `expm1` 还原。

写作重点：

- 与 Kaggle RMSLE 指标一致。
- 降低高价长尾对模型训练的影响。
- 让模型优化目标更稳定。

### 4.2 异常值处理

说明删除训练集中的已知异常样本，但不处理测试集。

建议对应代码：

- `house_price/data.py`

### 4.3 缺失值处理

按变量含义分组说明：

- 设施不存在类缺失：填充为 `None` 或 `0`
- 数值型缺失：按中位数或特定语义填充
- 类别型缺失：填充为 `None` 或众数

建议对应代码：

- `house_price/preprocessing.py`
- `house_price/advanced_preprocessing.py`

### 4.4 偏态数值变量处理

说明对偏态较强的数值变量进行 log 或 Box-Cox/Yeo-Johnson 类变换，以降低极端值影响。

建议引用：

- `reports/eda/tables/numeric_skewness.csv`
- `house_price/advanced_preprocessing.py`

### 4.5 类别变量编码

说明使用了：

- 有序类别编码：保留质量等级顺序。
- One-hot 编码：处理普通无序类别。
- Target Encoding：在后期实验中捕捉类别分组的价格水平。

## 5. 特征工程

这一章要说明我们不是只把原始表喂给模型，而是根据房屋领域含义构造了新特征。

### 5.1 面积类组合特征

可能包括：

- 总居住面积
- 总地下室面积
- 总门廊面积
- 地上面积与地下室面积组合

写作重点：房屋面积直接影响价格，组合面积能比单列变量更完整描述房屋规模。

### 5.2 质量类组合特征

可能包括：

- 整体质量和整体状况组合
- 外部质量、厨房质量、地下室质量等综合质量信息

写作重点：质量等级是房价最强信号之一，组合后可以捕捉“面积大但质量差”或“面积一般但质量高”的差异。

### 5.3 年份和房龄特征

可能包括：

- 房屋年龄
- 距离翻新的年数
- 是否翻新

写作重点：房屋新旧程度影响价格，年份差值比原始年份更接近业务含义。

### 5.4 设施存在性特征

例如：

- 是否有地下室
- 是否有车库
- 是否有壁炉
- 是否有泳池

写作重点：缺失值在本数据集中常常意味着设施不存在，因此可以显式构造二值特征。

### 5.5 Target Encoding 特征

说明本项目后期使用 OOF Target Encoding，避免目标泄漏。

重点写：

- 对训练集使用 OOF 方式生成编码。
- 对测试集使用全量训练集统计并做平滑。
- 主要用于 `Neighborhood`、`MSSubClass`、`SaleType` 以及若干组合类别。
- 最终发现 TE 单独提升有限，但与上一轮最好方案融合后有增量。

建议引用：

- `house_price/target_encoding.py`
- `reports/modeling/20260507_target_encoding_experiment_report.md`

## 6. 建模方法

### 6.1 基线模型

说明最初使用基础预处理和若干常见回归模型，得到 Public Score `0.12859`。

建议引用：

- `main.py`
- `outputs/cv_results.csv`
- `experiments/experiment_log.csv`

### 6.2 线性模型

包括：

- Ridge
- Lasso
- ElasticNet

写作重点：高维 one-hot 特征下，正则化线性模型通常非常稳健。

### 6.3 核方法与支持向量回归

包括：

- Kernel Ridge
- SVR

写作重点：这类模型能捕捉一定非线性关系，并在小数据集上表现较好。

### 6.4 集成树模型

包括：

- XGBoost
- LightGBM
- CatBoost

写作重点：树模型适合处理非线性和特征交互，但也容易受小数据和高价尾部影响。

### 6.5 模型融合

说明使用了：

- 简单平均融合
- OOF 优化权重融合
- 保守融合
- 与当前最好文件的 50/50 log 融合

写作重点：融合的作用是降低单模型误差和方差，尤其在不同模型误差互补时效果明显。

## 7. 实验过程与结果分析

本章以 Kaggle Public Score 作为判断方案好坏的标准。本地 CV 只作为训练阶段参考，不作为最终结论依据。

### 7.1 Baseline

说明第一版 baseline：

- 基础预处理
- 基础模型融合
- Public Score：`0.12859`

### 7.2 高级特征与多模型融合

说明第二阶段：

- 高级缺失值处理
- 偏态修正
- 更多模型
- OOF 融合
- 裁剪前后对比

关键结果：

- `20260506_optimized_weight_blend.csv`：`0.12173`
- `20260506_optimized_weight_blend_clipped.csv`：`0.11865`

### 7.3 高价尾部校准

说明第三阶段：

- 比较不同裁剪分位数
- 比较硬裁剪与软收缩
- 发现 q993 硬裁剪最稳

关键结果：

- `20260507_opt_clip_q993.csv`：`0.11805`

### 7.4 Target Encoding 增量实验

说明第四阶段：

- 新增 OOF Target Encoding 特征
- 训练 TE 模型族
- 与当前最好文件做 log 融合

关键结果：

- `20260507_te_conservative_blend_clip_q993.csv`：`0.11802`
- `20260507_te_weighted_blend_mix_current_best_clip_q993.csv`：`0.11774`
- `20260507_te_simple_blend_mix_current_best_clip_q993.csv`：`0.11758`

### 7.5 实验结果总表

最终报告中建议放一张简化表：

| 阶段 | 方案 | Public Score | 主要结论 |
| --- | --- | --- | --- |
| Baseline | 基础融合 | 0.12859 | 跑通完整流程 |
| 高级融合 | optimized blend | 0.12173 | 高级特征有效，但高价尾部偏高 |
| 高级融合 + 裁剪 | optimized clipped | 0.11865 | 裁剪显著改善 |
| 高价校准 | q993 clipping | 0.11805 | q993 是较优高价控制点 |
| TE 增量融合 | TE simple mix | 0.11758 | 当前最好，TE 提供互补信息 |

## 8. 最终方案

### 8.1 最终提交文件

最终文件：

`20260507_te_simple_blend_mix_current_best_clip_q993.csv`

Public Score：

`0.11758`

### 8.2 最终方案组成

最终方案由以下部分组成：

- 高级数据预处理
- 面积、质量、年份、设施存在性等人工特征
- 多模型 OOF 融合
- q993 高价硬裁剪
- OOF Target Encoding 保守融合
- 与上一轮最优提交进行 50/50 log 融合

### 8.3 为什么可以作为阶段最终结果

说明：

- 分数已进入前 100 左右，结果较好。
- 继续微调裁剪比例收益很小。
- 排行榜高分存在外部数据或泄漏可能，不适合作为正常建模目标。
- 当前方案具有完整实验链路和可解释性，适合作为期末报告最终结果。

## 9. 总结与反思

### 9.1 有效经验

- 目标变量 log 变换非常重要。
- 缺失值应结合业务含义处理。
- 质量、面积、地理位置是最关键的房价因素。
- 高价尾部风险会显著影响提交分数。
- 单纯追求本地 CV 可能误导。
- 模型融合尤其是互补模型融合比单模型调参更有效。

### 9.2 局限性

- 训练数据较小，验证波动不可避免。
- Public leaderboard 只代表测试集的一部分。
- 未使用外部数据。
- 未继续进行更复杂的 stacking 或原生类别 CatBoost 方案。

### 9.3 后续改进方向

- CatBoost 原生类别特征建模。
- 更系统的特征选择。
- 多层 stacking。
- 更稳健的高价样本建模。
- 私榜结果验证后再判断泛化能力。

## 10. 附录建议

最终报告可附：

- 主要代码文件说明
- 重要图表索引
- 实验记录表
- 最终提交文件说明

建议引用：

- `NAVIGATION.md`
- `reports/README.md`
- `submissions/README.md`
- `experiments/experiment_log.csv`

## 11. 后续写作拆分计划

为了避免一次性生成过长报告，后续建议按以下顺序逐章补写：

1. `01_project_background.md`：项目背景、任务说明、评价指标。
2. `02_data_understanding_and_eda.md`：数据集说明与 EDA 正文。
3. `03_preprocessing_and_feature_engineering.md`：预处理与特征工程。
4. `04_modeling_methods.md`：模型和融合方法。
5. `05_experiment_results.md`：实验过程与结果分析。
6. `06_final_solution_and_reflection.md`：最终方案、总结与反思。
7. `final_report.md`：合并成最终报告。
