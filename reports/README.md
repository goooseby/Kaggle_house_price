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
