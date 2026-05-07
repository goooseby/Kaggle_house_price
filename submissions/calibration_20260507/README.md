# 20260507 高价校准候选提交

这个目录保存“高价校准 + 保守融合”主题生成的候选提交文件。

生成命令：

```powershell
conda run -n kaggle_house python scripts/generate_calibration_candidates.py
```

本轮没有重新训练底层模型，而是基于上一轮已经生成的预测文件做：

- 不同分位数上限裁剪
- 高价尾部软压缩
- optimized/simple/inverse 的保守 log 空间融合

推荐优先提交：

1. `20260507_opt_clip_q995.csv`
2. `20260507_mix50_opt_simple_clip_q997.csv`
3. `20260507_opt_soft_q995_s035.csv`
4. `20260507_mean_opt_simple_inverse_clip_q997.csv`

分析报告：

- `reports/20260507_calibration_round_report.md`

辅助统计：

- `candidate_summary.csv`
- `high_price_impact.csv`
