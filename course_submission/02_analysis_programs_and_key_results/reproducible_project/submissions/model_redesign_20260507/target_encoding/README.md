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

最终以 Kaggle Public Score 判断好坏。补充提交后，`simple` 融合反而优于 `weighted` 和 `conservative`；更稳定的选择是 `clip_q993` 版本，以及它们与上一轮最好方案的 `mix_current_best` 版本。

## Public Score 反馈

| 文件 | Public Score |
| --- | --- |
| `20260507_te_simple_blend_mix_current_best_clip_q993.csv` | `0.11758` |
| `20260507_te_conservative_blend_mix_current_best_clip_q993.csv` | `0.11765` |
| `20260507_te_weighted_blend_mix_current_best_clip_q993.csv` | `0.11774` |
| `20260507_te_simple_blend_clip_q993.csv` | `0.11794` |
| `20260507_te_conservative_blend_clip_q993.csv` | `0.11802` |
| `20260507_te_weighted_blend_clip_q993.csv` | `0.11819` |
| `20260507_te_conservative_blend.csv` | `0.12044` |
| `20260507_te_weighted_blend.csv` | `0.12076` |

结论：未裁剪 TE 版本明显受高价尾部影响；q993 裁剪后，纯 TE simple blend 已略优于上一轮 q993 最好方案。TE 与上一轮最好文件做 50/50 log 融合后带来了明确增量，其中 simple blend mix 得到当前最好 Public Score `0.11758`。
