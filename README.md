# Kaggle House Price

这是 Kaggle 入门题 **House Prices - Advanced Regression Techniques** 的本地实验项目。

当前最好 Kaggle Public Score：`0.11765`  
当前最好文件：`20260507_te_conservative_blend_mix_current_best_clip_q993.csv`

## 当前状态

项目已经完成以下几轮工作：

- baseline 建模与第一次提交：Public Score `0.12859`
- 高级特征工程、OOF 融合与裁剪校准：最好到 `0.11805`
- EDA 补充：完整报告、图表、统计表已落盘
- 本地 CV 与 Public Score 差异审计：确定后续不能只看普通 CV
- Target Encoding 模型重设计实验：当前最好到 `0.11765`

最新结论：Target Encoding 单独替代旧方案并不明显，但它和当前最好方案做 50/50 log 融合后带来了有效增量。

## 快速导航

- [项目总导航](NAVIGATION.md)
- [报告总索引](reports/README.md)
- [提交文件索引](submissions/README.md)
- [实验记录](experiments/experiment_log.csv)
- [EDA 图表导览](reports/eda/20260506_eda_visual_guide.md)
- [模型重设计方案](reports/modeling/20260507_model_redesign_plan.md)
- [Target Encoding 实验报告](reports/modeling/20260507_target_encoding_experiment_report.md)

## 常用命令

运行当前 Target Encoding 实验：

```powershell
conda run -n kaggle_house python run_target_encoding_experiment.py
```

重新生成完整 EDA：

```powershell
conda run -n kaggle_house python scripts/generate_full_eda_report.py
```

重新审计本地验证与 Public Score：

```powershell
conda run -n kaggle_house python scripts/audit_validation_metrics.py
```

## 目录说明

```text
house_price/       核心代码模块
scripts/           分析、审计、候选生成脚本
experiments/       实验中间结果、评分面板、OOF 预测
reports/           人工整理后的报告和复盘
reports/eda/       EDA 报告、图表、统计表
submissions/       Kaggle 候选提交文件和说明
docs/              早期项目文档
outputs/           自动生成产物，默认不提交 Git
```

## 当前提交候选重点

Target Encoding 本轮已提交并验证：

| 文件 | Public Score | 说明 |
| --- | --- | --- |
| `20260507_te_conservative_blend_mix_current_best_clip_q993.csv` | `0.11765` | 当前最好，TE 保守融合与旧最好方案 50/50 log 融合 |
| `20260507_te_weighted_blend_mix_current_best_clip_q993.csv` | `0.11774` | TE 优化权重融合与旧最好方案 50/50 log 融合 |
| `20260507_te_conservative_blend_clip_q993.csv` | `0.11802` | 纯 TE 保守融合 q993 裁剪 |
| `20260507_te_weighted_blend_clip_q993.csv` | `0.11819` | 纯 TE 优化权重融合 q993 裁剪 |

## 下一步方向

下一轮不建议继续微调裁剪比例。更高价值方向是扩大模型差异性，例如 CatBoost 原生类别特征、不同特征空间的模型族融合、稳健 stacking，以及继续用新评分面板约束高价尾部风险。
