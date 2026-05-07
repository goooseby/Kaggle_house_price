# Target Encoding 实验产物导航

本目录记录 20260507 新模型重设计中的第一轮正式主题：OOF Target Encoding。

## 文件说明

- `score_panel.csv`：本轮所有单模型与融合模型的新评分面板。
- `oof_predictions.csv`：训练集 OOF log 预测，用于后续融合、诊断和评分校准。
- `test_log_predictions.csv`：测试集 log 预测，用于生成提交文件。

## 对应报告

- `reports/modeling/20260507_target_encoding_experiment_report.md`

## 复现实验

```powershell
conda run -n kaggle_house python run_target_encoding_experiment.py
```