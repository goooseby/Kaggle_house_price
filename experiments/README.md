# Experiments

这个目录用于记录每一次正式实验，而不是保存所有临时输出。

原则：

- 每个实验必须有清楚的名字。
- 每个实验必须记录本地 CV。
- 如果提交到 Kaggle，必须补充 Public Score。
- 不只看分数，也要记录关键改动，否则后续无法判断是哪一步有效。

## 命名建议

实验名使用：

```text
YYYYMMDD_short_description
```

例如：

```text
20260506_baseline_blend
20260507_full_optimization_round
```

## 当前基线

当前已知基线：

| experiment | local_cv_rmse | kaggle_public_score | note |
| --- | ---: | ---: | --- |
| 20260506_baseline_blend | 0.11122 | 0.12859 | First working baseline, public rank around top 30%. |

后续实验应和这个基线比较。
