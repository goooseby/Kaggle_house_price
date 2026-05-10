# Target Encoding 提交候选导航

本目录保存由 `run_target_encoding_experiment.py` 生成的提交候选文件。

## 命名规则

- `20260507_te_simple_blend.csv`：本轮 TE 单纯等权融合。
- `20260507_te_weighted_blend.csv`：本轮 TE OOF 优化权重融合。
- `20260507_te_conservative_blend.csv`：本轮 TE 保守融合。
- `*_clip_q993.csv`：在对应融合基础上做 q993 高价硬裁剪。
- `*_mix_current_best_clip_q993.csv`：对应 q993 裁剪版本与当前最好提交做 50/50 log 融合。

## 本次生成文件

| candidate | path |
| --- | --- |
| te_simple_blend | 20260507_te_simple_blend.csv |
| te_simple_blend_clip_q993 | 20260507_te_simple_blend_clip_q993.csv |
| te_simple_blend_mix_current_best_clip_q993 | 20260507_te_simple_blend_mix_current_best_clip_q993.csv |
| te_weighted_blend | 20260507_te_weighted_blend.csv |
| te_weighted_blend_clip_q993 | 20260507_te_weighted_blend_clip_q993.csv |
| te_weighted_blend_mix_current_best_clip_q993 | 20260507_te_weighted_blend_mix_current_best_clip_q993.csv |
| te_conservative_blend | 20260507_te_conservative_blend.csv |
| te_conservative_blend_clip_q993 | 20260507_te_conservative_blend_clip_q993.csv |
| te_conservative_blend_mix_current_best_clip_q993 | 20260507_te_conservative_blend_mix_current_best_clip_q993.csv |

## 选择原则

最终以 Kaggle Public Score 判断好坏；提交时优先考虑 `conservative`、`weighted`、`clip_q993` 和 `mix_current_best` 这些更稳的版本。

## Public Score 反馈

| 文件 | Public Score |
| --- | --- |
| `20260507_te_conservative_blend_mix_current_best_clip_q993.csv` | `0.11765` |
| `20260507_te_weighted_blend_mix_current_best_clip_q993.csv` | `0.11774` |
| `20260507_te_conservative_blend_clip_q993.csv` | `0.11802` |
| `20260507_te_weighted_blend_clip_q993.csv` | `0.11819` |

结论：TE 纯模型提升有限，但与上一轮最好文件做 50/50 log 融合后带来了明确增量。
