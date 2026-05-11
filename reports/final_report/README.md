# 期末报告写作目录

本目录用于逐步整理最终提交用的数据挖掘期末大作业报告。

## 当前文件

- [00_report_outline.md](00_report_outline.md)：完整报告大纲和写作计划。
- [01_project_background.md](01_project_background.md)：项目背景、任务说明、评价指标和整体流程。
- [02_dataset_description.md](02_dataset_description.md)：数据来源、数据规模、特征类型、缺失值和目标变量概况。
- [03_data_understanding_and_eda.md](03_data_understanding_and_eda.md)：探索性数据分析正文。
- [04_preprocessing.md](04_preprocessing.md)：数据预处理正文。
- [05_feature_engineering.md](05_feature_engineering.md)：特征工程正文。
- [06_modeling_methods.md](06_modeling_methods.md)：建模方法正文。
- [07_experiment_results.md](07_experiment_results.md)：实验过程、结果对比和迭代分析。
- [08_final_solution.md](08_final_solution.md)：最终方案。
- [09_summary_and_reflection.md](09_summary_and_reflection.md)：总结与反思。
- [final_report.md](final_report.md)：最终合并版报告。

## 已有材料来源

- EDA 报告：`reports/eda/20260506_full_eda_report.md`
- EDA 图表导览：`reports/eda/20260506_eda_visual_guide.md`
- Target Encoding 实验：`reports/modeling/20260507_target_encoding_experiment_report.md`
- 实验记录：`experiments/experiment_log.csv`

## 本目录补充表格

- `tables/dataset_profile.csv`：训练集、测试集、提交样例的数据规模。
- `tables/feature_type_summary.csv`：原始特征数据类型统计。
- `tables/feature_domain_groups.csv`：按业务含义划分的特征组。
- `tables/missing_overview.csv`：训练集和测试集缺失值概况。
- `tables/target_summary_for_report.csv`：目标变量原始尺度与 log 尺度统计。
- `tables/preprocessing_feature_summary.csv`：预处理后特征规模摘要。
- `tables/skew_transformed_columns.csv`：执行偏态修正的数值列清单。
- `tables/experiment_results_summary.csv`：关键提交结果与分数提升摘要。

## 本目录补充图表

- `figures/public_score_progress.png`：关键阶段 Public Score 迭代折线图。
- `figures/score_improvement_by_stage.png`：关键操作带来的分数改善柱状图。
- `figures/submitted_candidate_ranking.png`：所有已提交候选的 Public Score 排名图。
- `figures/clipping_threshold_curve.png`：高价裁剪分位数校准曲线。
- `figures/target_encoding_candidate_comparison.png`：Target Encoding 候选方案对比图。

重新生成命令：

```powershell
conda run -n kaggle_house python scripts/generate_final_report_figures.py
```
