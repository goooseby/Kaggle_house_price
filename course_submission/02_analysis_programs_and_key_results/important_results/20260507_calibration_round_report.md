# 20260507 高价校准与保守融合实验报告

## 1. 本轮实验目的

上一轮 Kaggle 提交结果显示，`clipped` 版本明显优于未裁剪版本：

| 对比 | 未裁剪 | 裁剪后 | 提升 |
| --- | ---: | ---: | ---: |
| optimized_weight_blend | 0.12173 | 0.11865 | 0.00308 |
| ridge_stack | 0.12258 | 0.11982 | 0.00276 |

这说明高价预测尾部对 Public Score 影响很大。因此本轮不重新训练底层模型，而是专门做：

1. 高价预测上限校准。
2. 高价尾部软压缩。
3. 更保守的 log 空间融合。

本轮要回答的问题是：

- q997 裁剪是否已经是最优，还是更低/更高的上限更好？
- optimized 和 simple 两种融合是否可以通过保守混合提升线上稳定性？
- 软压缩是否比硬裁剪更适合高价尾部？

## 2. 本轮输入文件

本轮所有候选都基于上一轮已经生成的提交文件，不重新训练模型。

| 简称 | 输入文件 | 上一轮 Public Score | 说明 |
| --- | --- | ---: | --- |
| `opt` | `submissions/20260506_optimized_weight_blend.csv` | 0.12173 | 非负权重优化融合，未裁剪 |
| `simple` | `submissions/20260506_simple_blend.csv` | 0.12153 | 7 模型简单平均，未裁剪 |
| `inverse` | `submissions/20260506_inverse_cv_blend.csv` | 未提交 | 按 CV 反比加权，未裁剪 |
| `stack` | `submissions/20260506_ridge_stack.csv` | 0.12258 | Ridge stacking，未裁剪 |
| `previous_best` | `submissions/20260506_optimized_weight_blend_clipped.csv` | 0.11865 | 当前最好结果，作为对比基准 |

这些文件都已经是 Kaggle 提交格式：

```text
Id,SalePrice
```

本轮脚本读取这些 `SalePrice` 预测，生成新的后处理候选。

## 3. 本轮生成脚本

脚本路径：

```text
scripts/generate_calibration_candidates.py
```

运行命令：

```powershell
conda run -n kaggle_house python scripts/generate_calibration_candidates.py
```

输出目录：

```text
submissions/calibration_20260507/
```

辅助输出：

| 文件 | 作用 |
| --- | --- |
| `candidate_summary.csv` | 所有候选文件的预测分布摘要 |
| `high_price_impact.csv` | 上一轮高价预测样本在不同方案下的变化 |
| `README.md` | 本轮候选提交说明 |

## 4. 裁剪阈值来源

本轮裁剪阈值全部来自训练集 `SalePrice` 分位数。

| 阈值名称 | 分位数 | 房价上限 |
| --- | ---: | ---: |
| `q990` | 99.0% | 442567.01 |
| `q993` | 99.3% | 473189.50 |
| `q995` | 99.5% | 527331.92 |
| `q997` | 99.7% | 572998.30 |
| `q999` | 99.9% | 689920.00 |

上一轮最优 `20260506_optimized_weight_blend_clipped.csv` 使用的上限接近 `q997`。

本轮重点测试：

- 更强裁剪：`q990`、`q993`、`q995`
- 原有强度附近：`q997`
- 更宽松裁剪：`q999`
- 软压缩：超过阈值后不直接截断，而是按比例压缩尾部

## 5. 方案一：单一 optimized 预测的不同硬裁剪

这组方案只使用 `opt` 作为输入，也就是：

```text
20260506_optimized_weight_blend.csv
```

然后对 `SalePrice` 做不同上限裁剪。

公式：

```text
new_prediction = min(original_prediction, threshold)
```

| 方案名 | 输出文件 | 做了什么 | 最大预测值 | 目的 |
| --- | --- | --- | ---: | --- |
| `opt_clip_q990` | `20260507_opt_clip_q990.csv` | opt 预测裁剪到 q990 | 442567.01 | 测试非常强的高价压制 |
| `opt_clip_q993` | `20260507_opt_clip_q993.csv` | opt 预测裁剪到 q993 | 473189.50 | 测试较强高价压制 |
| `opt_clip_q995` | `20260507_opt_clip_q995.csv` | opt 预测裁剪到 q995 | 527331.92 | 推荐提交，测试比上一轮更强的裁剪 |
| `opt_clip_q997` | `20260507_opt_clip_q997.csv` | opt 预测裁剪到 q997 | 572998.30 | 复现上一轮 clipped 附近强度 |
| `opt_clip_q999` | `20260507_opt_clip_q999.csv` | opt 预测裁剪到 q999 | 689920.00 | 测试更宽松裁剪 |

这组方案的意义：

- 如果 `q995` 比上一轮 `q997` 更好，说明高价上限还可以继续下降。
- 如果 `q990/q993` 更好，说明 Public 集合里高价预测整体偏高。
- 如果 `q999` 更好，说明上一轮裁剪可能过强。

## 6. 方案二：optimized 与 simple 的保守融合后裁剪

这组方案使用两个上一轮较重要的非 stacking 预测：

```text
opt    = 20260506_optimized_weight_blend.csv
simple = 20260506_simple_blend.csv
```

融合在 log 空间进行：

```text
new_prediction = expm1(w1 * log1p(opt) + w2 * log1p(simple))
```

之后统一做 `q997` 裁剪。

| 方案名 | 输出文件 | 权重 | 最大预测值 | 目的 |
| --- | --- | --- | ---: | --- |
| `mix30_opt70_simple_clip_q997` | `20260507_mix30_opt70_simple_clip_q997.csv` | 30% opt + 70% simple | 572998.30 | 更偏向 simple，测试稳健融合 |
| `mix50_opt_simple_clip_q997` | `20260507_mix50_opt_simple_clip_q997.csv` | 50% opt + 50% simple | 572998.30 | 推荐提交，测试中性保守融合 |
| `mix70_opt30_simple_clip_q997` | `20260507_mix70_opt30_simple_clip_q997.csv` | 70% opt + 30% simple | 572998.30 | 更偏向 opt，测试轻度保守化 |

这组方案的意义：

- 上一轮 `simple_blend` 未裁剪版本线上略优于 `optimized_weight_blend`。
- 说明 `simple` 可能更稳。
- 所以本轮尝试把 `optimized` 往 `simple` 拉回一点。

如果这组方案优于 `0.11865`，下一步应继续研究保守融合权重。

## 7. 方案三：多个非 stacking 预测的保守平均

这组方案把多个非 stacking 或低风险预测做 log 空间平均，再做 q997 裁剪。

### 7.1 opt + simple + inverse

| 方案名 | 输出文件 | 输入 | 最大预测值 | 目的 |
| --- | --- | --- | ---: | --- |
| `mean_opt_simple_inverse_clip_q997` | `20260507_mean_opt_simple_inverse_clip_q997.csv` | opt / simple / inverse 各 1/3 | 572998.30 | 推荐提交，测试三种非 stacking 融合是否更稳 |

这组不使用 `ridge_stack`，避免 stacking 偏乐观问题。

### 7.2 opt + simple + stack

| 方案名 | 输出文件 | 输入 | 最大预测值 | 目的 |
| --- | --- | --- | ---: | --- |
| `mean_opt_simple_stack_clip_q997` | `20260507_mean_opt_simple_stack_clip_q997.csv` | opt / simple / stack 各 1/3 | 572998.30 | 测试 stack 是否在保守平均后仍有价值 |

注意：

`ridge_stack` 上一轮线上表现不好，因此这个文件不是优先提交，只作为“stack 是否还有信息价值”的备选。

## 8. 方案四：高价尾部软压缩

硬裁剪会把超过阈值的预测全部压到同一个上限，可能过于粗暴。

软压缩的公式是：

```text
if prediction > threshold:
    new_prediction = threshold + (prediction - threshold) * tail_strength
else:
    new_prediction = prediction
```

其中 `tail_strength` 越小，压缩越强。

| 方案名 | 输出文件 | 输入 | 阈值 | 尾部保留比例 | 最大预测值 | 目的 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `opt_soft_q995_s035` | `20260507_opt_soft_q995_s035.csv` | opt | q995 | 35% | 643039.99 | 推荐提交，测试软压缩是否优于硬裁剪 |
| `opt_soft_q997_s050` | `20260507_opt_soft_q997_s050.csv` | opt | q997 | 50% | 715462.36 | 测试更宽松软压缩 |
| `mix50_soft_q995_s035` | `20260507_mix50_soft_q995_s035.csv` | 50% opt + 50% simple | q995 | 35% | 645326.43 | 测试保守融合 + 软压缩 |

这组方案的意义：

- 如果软压缩优于硬裁剪，说明高价房并不应该被完全截断。
- 如果软压缩差于硬裁剪，说明当前模型高价尾部仍然偏高，硬裁剪更安全。

## 9. 所有候选文件汇总

| 方案 | 输出文件 | 类型 | 最大预测值 | 与上一轮最优平均差异 |
| --- | --- | --- | ---: | ---: |
| `opt_clip_q990` | `20260507_opt_clip_q990.csv` | opt 硬裁剪 | 442567.01 | 607.08 |
| `opt_clip_q993` | `20260507_opt_clip_q993.csv` | opt 硬裁剪 | 473189.50 | 304.66 |
| `opt_clip_q995` | `20260507_opt_clip_q995.csv` | opt 硬裁剪 | 527331.92 | 62.82 |
| `opt_clip_q997` | `20260507_opt_clip_q997.csv` | opt 硬裁剪 | 572998.30 | 0.22 |
| `opt_clip_q999` | `20260507_opt_clip_q999.csv` | opt 硬裁剪 | 689920.00 | 86.22 |
| `mix30_opt70_simple_clip_q997` | `20260507_mix30_opt70_simple_clip_q997.csv` | opt/simple log 融合 | 572998.30 | 638.52 |
| `mix50_opt_simple_clip_q997` | `20260507_mix50_opt_simple_clip_q997.csv` | opt/simple log 融合 | 572998.30 | 456.29 |
| `mix70_opt30_simple_clip_q997` | `20260507_mix70_opt30_simple_clip_q997.csv` | opt/simple log 融合 | 572998.30 | 273.94 |
| `mean_opt_simple_inverse_clip_q997` | `20260507_mean_opt_simple_inverse_clip_q997.csv` | 三模型保守平均 | 572998.30 | 624.42 |
| `mean_opt_simple_stack_clip_q997` | `20260507_mean_opt_simple_stack_clip_q997.csv` | 含 stack 的保守平均 | 572998.30 | 410.40 |
| `opt_soft_q995_s035` | `20260507_opt_soft_q995_s035.csv` | opt 软压缩 | 643039.99 | 66.15 |
| `opt_soft_q997_s050` | `20260507_opt_soft_q997_s050.csv` | opt 软压缩 | 715462.36 | 100.58 |
| `mix50_soft_q995_s035` | `20260507_mix50_soft_q995_s035.csv` | opt/simple 融合 + 软压缩 | 645326.43 | 523.93 |

## 10. 推荐提交顺序

本轮推荐先提交 4 个文件：

| 顺序 | 文件 | 为什么提交 |
| ---: | --- | --- |
| 1 | `20260507_opt_clip_q995.csv` | 直接验证 q995 是否比上一轮 q997 更好，是最干净的高价上限实验 |
| 2 | `20260507_mix50_opt_simple_clip_q997.csv` | 验证 simple 的稳健性是否能改善 optimized |
| 3 | `20260507_opt_soft_q995_s035.csv` | 验证软压缩是否比硬裁剪更自然 |
| 4 | `20260507_mean_opt_simple_inverse_clip_q997.csv` | 验证三个非 stacking 融合是否比单一 optimized 更稳 |

当前需要打败的最好成绩：

```text
0.11865
```

## 11. 高价样本影响

完整高价样本影响表：

```text
submissions/calibration_20260507/high_price_impact.csv
```

关键样本仍然是：

```text
Id = 2550
```

它在上一轮 optimized 中预测约：

```text
857926
```

上一轮 best clipped 把它压到：

```text
573156
```

本轮不同方案会进一步测试：

- 压到 `527331` 是否更好
- 保留到 `643039` 左右是否更好
- 与 simple 融合后是否更稳

这个样本仍然是判断高价校准是否有效的核心观察点。

## 12. 提交后如何判断

如果 `opt_clip_q995` 优于 `0.11865`：

- 高价上限还可以继续降低。
- 下一轮继续测试 q992/q994/q996 等细分阈值。

如果 `mix50_opt_simple_clip_q997` 优于 `0.11865`：

- 保守融合比单一 optimized 更可靠。
- 下一轮继续搜索 opt/simple/inverse 的融合比例。

如果 `opt_soft_q995_s035` 优于 `0.11865`：

- 高价尾部不应该硬截断，软压缩更合适。
- 下一轮继续调 `tail_strength`。

如果所有候选都没有超过 `0.11865`：

- 上一轮 q997 硬裁剪可能已经接近当前模型后处理上限。
- 下一步应转向 OOF target encoding、特征修正或更严格 stacking。

## 13. 本轮结论

本轮产物不是为了“一次性全交”，而是为了验证三个具体方向：

1. 更低高价上限是否有效。
2. 保守融合是否有效。
3. 软压缩是否有效。

这些方向一旦通过 Kaggle Public Score 验证，就可以继续精细化；如果不通过，就避免在错误方向上继续投入。
