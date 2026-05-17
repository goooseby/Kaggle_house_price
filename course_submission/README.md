# 数据挖掘课程提交材料目录

本目录按课程要求整理最终提交材料。PPT 和 Word 报告由组内后续放入对应目录，其余数据、程序和重要结果已经整理完成。

## 目录结构

1. `01_raw_data/`
   - Kaggle House Prices 原始数据。
   - 包含 `train.csv`、`test.csv`、`sample_submission.csv` 和 `data_description.txt`。

2. `02_analysis_programs_and_key_results/`
   - 分析程序集及重要结果。
   - `reproducible_project/` 保留一份可以直接运行的项目主线代码和必要数据。
   - `important_results/` 保留最终报告、关键提交文件、结果表、图表和排行榜截图。

3. `03_presentation_ppt/`
   - 放置演示 PPT。
   - 建议命名：`DM1班第1组房价预测.pptx`，按实际班级和小组序号修改。

4. `04_case_report_docx/`
   - 放置数据挖掘案例分析报告 Word 版。
   - 建议命名：`DM1班第1组房价预测.docx`，按实际班级和小组序号修改。

## 最终结果

- 最终最好提交文件：`20260507_te_simple_blend_mix_current_best_clip_q993.csv`
- Kaggle Public Score：`0.11758`
- 最终方案：高级特征工程、多模型融合、q993 高价裁剪、OOF Target Encoding 和 50/50 log 融合。

## 复现说明

进入 `02_analysis_programs_and_key_results/reproducible_project/` 后，可运行：

```powershell
python run_target_encoding_experiment.py
```

该命令会基于原始数据重新训练 Target Encoding 模型族，并生成最终阶段的候选提交文件。
