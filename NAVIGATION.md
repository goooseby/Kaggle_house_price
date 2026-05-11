# 项目导航

本项目围绕 Kaggle House Prices 入门题，目标是用清晰、可复现的实验流程逐步提升提交分数，同时保留报告材料。

当前最好 Public Score：`0.11758`  
当前最好提交：`20260507_te_simple_blend_mix_current_best_clip_q993.csv`

## 常用入口

- `main.py`：基础入口，保留早期基线流程。
- `run_optimization.py`：上一轮高级特征、模型融合与校准候选生成入口。
- `run_target_encoding_experiment.py`：当前正式实验入口，执行 OOF Target Encoding 并生成提交候选。

## 核心代码

- `house_price/preprocessing.py`：早期基础特征处理。
- `house_price/advanced_preprocessing.py`：当前主要高级特征工程。
- `house_price/modeling.py`：早期建模工具。
- `house_price/advanced_modeling.py`：上一轮高级模型与融合工具。
- `house_price/target_encoding.py`：当前新增的 OOF Target Encoding 特征。

## 实验目录

- `experiments/model_redesign_20260507/`：新一轮大改模型方案的实验根目录。
- `experiments/model_redesign_20260507/target_encoding/`：当前 Target Encoding 实验产物。

## 报告目录

- `reports/README.md`：报告总索引。
- `reports/final_report/final_report.md`：期末报告最终合并版。
- `reports/eda/`：完整 EDA 报告、图表、统计表。
- `reports/modeling/20260507_target_encoding_experiment_report.md`：当前 Target Encoding 实验报告。

## 提交候选目录

- `submissions/README.md`：提交文件说明总索引。
- `submissions/calibration_20260507/`：上一轮高价校准候选。
- `submissions/model_redesign_20260507/target_encoding/`：当前 Target Encoding 候选提交文件。

## 当前推荐复现命令

```powershell
conda run -n kaggle_house python run_target_encoding_experiment.py
```
