# EDA 输出说明

这个目录保存完整探索性分析的交付物。

生成命令：

```powershell
conda run -n kaggle_house python scripts/generate_full_eda_report.py
```

主要文件：

- `20260506_full_eda_report.md`：中文完整 EDA 报告
- `20260506_eda_visual_guide.md`：图表导览版报告，适合快速浏览
- `figures/`：报告中引用的图表 PNG
- `tables/`：报告中使用的统计表 CSV

这部分内容独立于建模优化流程，用于补充报告材料和解释建模决策。
