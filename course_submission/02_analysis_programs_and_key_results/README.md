# 分析程序集及重要结果

本目录用于提交课程要求中的“分析程序集及重要结果”。

## reproducible_project

`reproducible_project/` 是一份可运行的项目主线副本，包含：

- `house_price/`：核心代码模块，包括数据读取、预处理、特征工程、建模和 Target Encoding。
- `scripts/`：EDA、报告表格、报告图表和候选提交生成脚本。
- `run_optimization.py`：高级特征工程和多模型融合实验入口。
- `run_target_encoding_experiment.py`：最终 Target Encoding 实验入口，可生成最终最好候选相关文件。
- `house-prices-advanced-regression-techniques/`：运行所需原始数据副本。
- `submissions/`、`outputs/`、`experiments/`：已经生成的重要结果和中间记录。

推荐复现命令：

```powershell
cd course_submission\02_analysis_programs_and_key_results\reproducible_project
python run_target_encoding_experiment.py
```

## important_results

`important_results/` 是整理后的结果材料，适合写报告和 PPT 时直接引用：

- `final_report_markdown/`：最终 Markdown 报告及其图片、表格。
- `selected_submissions/`：关键提交文件，包括 baseline、裁剪方案和最终最好方案。
- `summary_tables/`：实验结果摘要、裁剪候选摘要、TE CV 结果等。
- `figures/`：最终报告第 7 章使用的核心结果图。
- `experiment_log.csv`：完整实验分数记录。
- `排名结果87-5030.png`：Kaggle 排行榜截图。

最终最好方案：

```text
selected_submissions/20260507_te_simple_blend_mix_current_best_clip_q993.csv
```

Public Score：

```text
0.11758
```
