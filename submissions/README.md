# Submissions

这个目录用于放准备提交到 Kaggle 的候选 CSV 文件，以及候选文件说明。

CSV 文件默认不提交到 Git，因为它们是生成物。提交前建议检查：

- 是否有 1459 行
- 是否只有 `Id` 和 `SalePrice` 两列
- `Id` 是否和 `sample_submission.csv` 顺序一致
- `SalePrice` 是否无缺失、无负数、无无穷值
- 预测最大值和最小值是否明显离谱

当前第一版提交文件还在：

```text
outputs/submission_baseline.csv
```

后续正式优化回合建议把候选提交复制或输出到：

```text
submissions/
```

例如：

```text
submissions/20260507_weighted_blend.csv
submissions/20260507_blend_clipped.csv
```

当前候选文件说明：

- [20260506_candidates.md](20260506_candidates.md)
- [calibration_20260507/README.md](calibration_20260507/README.md)
- [model_redesign_20260507/README.md](model_redesign_20260507/README.md)
