# Reports

这个目录用于保存人工整理的实验报告。

和 `outputs/` 的区别：

- `outputs/` 是脚本自动生成的运行产物，默认不进 Git。
- `reports/` 是我们整理后的结论、对比和复盘，可以进 Git。

下一轮优化完成后，建议在这里生成：

```text
reports/20260507_optimization_round_report.md
```

报告内容至少包括：

- 实验目标
- 使用的特征处理
- 使用的模型
- 调参方式
- 单模型 CV
- 融合 CV
- 候选提交文件
- 和 Kaggle Public Score 的对比

## 当前报告

- [20260506_optimization_round_report.md](20260506_optimization_round_report.md)：完整优化回合报告
- [20260507_calibration_round_report.md](20260507_calibration_round_report.md)：高价校准与保守融合实验报告
- [20260507_calibration_feedback_review.md](20260507_calibration_feedback_review.md)：高价校准提交反馈复盘
- [modeling/20260507_target_encoding_experiment_report.md](modeling/20260507_target_encoding_experiment_report.md)：Target Encoding 正式实验报告
- [final_report/00_report_outline.md](final_report/00_report_outline.md)：期末报告总大纲
- [final_report/01_project_background.md](final_report/01_project_background.md)：期末报告第 1 章，项目背景与任务说明
- [final_report/02_dataset_description.md](final_report/02_dataset_description.md)：期末报告第 2 章，数据集说明
- [final_report/03_data_understanding_and_eda.md](final_report/03_data_understanding_and_eda.md)：期末报告第 3 章，EDA 正文
- [final_report/04_preprocessing.md](final_report/04_preprocessing.md)：期末报告第 4 章，数据预处理
- [final_report/05_feature_engineering.md](final_report/05_feature_engineering.md)：期末报告第 5 章，特征工程
- [final_report/06_modeling_methods.md](final_report/06_modeling_methods.md)：期末报告第 6 章，建模方法
- [final_report/07_experiment_results.md](final_report/07_experiment_results.md)：期末报告第 7 章，实验过程与结果分析
- [final_report/08_final_solution.md](final_report/08_final_solution.md)：期末报告第 8 章，最终方案
- [final_report/09_summary_and_reflection.md](final_report/09_summary_and_reflection.md)：期末报告第 9 章，总结与反思
- [final_report/final_report.md](final_report/final_report.md)：期末报告最终合并版
- [eda/20260506_full_eda_report.md](eda/20260506_full_eda_report.md)：完整探索性分析报告
- [eda/20260506_eda_visual_guide.md](eda/20260506_eda_visual_guide.md)：EDA 图表导览报告

## EDA 资产

完整 EDA 的图表和统计表位于：

```text
reports/eda/
  figures/
  tables/
```

重新生成命令：

```powershell
conda run -n kaggle_house python scripts/generate_full_eda_report.py
```
