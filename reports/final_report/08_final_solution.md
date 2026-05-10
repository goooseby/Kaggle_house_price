# 8. 最终方案

## 8.1 最终提交文件

经过多轮实验对比后，本项目最终选择以下文件作为阶段最终提交：

```text
submissions/model_redesign_20260507/target_encoding/20260507_te_conservative_blend_mix_current_best_clip_q993.csv
```

该文件在 Kaggle Public Leaderboard 上取得的 Public Score 为：

```text
0.11765
```

这是当前项目中已经提交并获得反馈的最好成绩。相比初始 baseline 的 `0.12859`，最终方案累计提升 `0.01094`。从实验过程看，该成绩并不是单一模型或单一技巧带来的，而是由稳定的数据预处理、领域特征工程、多模型融合、高价预测裁剪、Target Encoding 和 log 空间融合共同作用得到的。

## 8.2 最终方案整体流程

最终方案可以概括为以下流程：

```text
原始数据
  -> 删除训练集已知异常点
  -> 高级缺失值处理与类别编码
  -> 面积、质量、年份、设施等人工特征工程
  -> 多模型训练与 OOF 融合
  -> q993 高价硬裁剪
  -> OOF Target Encoding 模型族训练
  -> TE 保守融合与上一轮最优结果 50/50 log 融合
  -> 生成 Kaggle 提交文件
```

其中，训练目标始终使用 `log1p(SalePrice)`。最终融合也在 log 价格尺度上进行，这与 Kaggle RMSLE 的评价逻辑一致。预测结果生成后再通过 `expm1` 还原到原始房价尺度。

## 8.3 数据预处理与基础特征

最终方案沿用了高级预处理流程。该流程主要包括：

1. 删除训练集中两个 `GrLivArea` 极大但 `SalePrice` 较低的经典异常点。
2. 将目标变量转换为 `log1p(SalePrice)`。
3. 按字段含义处理缺失值，例如无车库、无地下室、无泳池等缺失填充为 `None` 或 0。
4. 为原始缺失字段构造 `_WasMissing` 缺失指示器。
5. 对质量类和等级类变量进行有序编码。
6. 对普通无序类别变量进行 one-hot 编码。
7. 对右偏数值变量进行 `log1p` 偏态修正。
8. 使用 `RobustScaler` 对数值矩阵进行稳健缩放。

这些预处理步骤保证了训练集和测试集使用一致的特征空间，也降低了异常值和偏态分布对模型训练的影响。

## 8.4 人工特征工程

最终方案中的人工特征主要围绕房价形成逻辑构造。核心特征类型包括：

- 面积组合特征，例如 `TotalSF`、`TotalLivingSF`、`TotalFinishedSF`、`TotalPorchSF`。
- 卫浴综合特征，例如 `TotalBathrooms`。
- 年份和房龄特征，例如 `HouseAge`、`RemodAge`、`GarageAge`、`IsNewHouse`、`IsRemodeled`。
- 设施存在性特征，例如 `HasGarage`、`HasBsmt`、`HasFireplace`、`HasPool`。
- 比例特征，例如 `FinishedBasementRatio`、`GarageAreaPerCar`、`SecondFloorRatio`。
- 质量交互特征，例如 `OverallQual_TotalSF`、`OverallQual_GrLivArea`、`OverallQual_GarageCars`。

这些特征使模型能够更直接地利用面积、质量、年份、设施和位置之间的关系。实验结果显示，高级特征工程和多模型融合使 Public Score 从 baseline 的 `0.12859` 提升到 `0.12173`，是最主要的提升来源之一。

## 8.5 多模型融合

最终方案不是依赖单一模型，而是使用多模型融合来提高稳定性。项目中使用过的模型包括：

- 正则化线性模型：Ridge、Lasso、ElasticNet。
- 核方法和支持向量回归：Kernel Ridge、SVR。
- 集成树模型：XGBoost、LightGBM、CatBoost。

多模型融合主要使用 OOF 预测作为基础。实验中比较了简单平均融合、非负权重优化融合、反比分数加权融合和 Ridge stacking。最终主线中更稳定的是非负权重优化融合与保守融合，而不是过度依赖 stacking。

这一阶段的意义在于：不同模型对线性关系、非线性关系、稀疏 one-hot 特征和树结构交互的敏感性不同，融合能够利用它们的误差互补。虽然本地 CV 只作为训练参考，但 Kaggle 反馈也表明，多模型融合明显优于最初的简单 baseline。

## 8.6 高价预测裁剪

高价裁剪是最终方案中的关键后处理步骤。实验发现，部分模型会对测试集中的高价房产生过高预测，从而影响 Kaggle RMSLE 分数。因此，项目对预测房价设置上限，限制极端高价外推。

最初的 `optimized_weight_blend` 未裁剪版本 Public Score 为：

```text
0.12173
```

加入裁剪后，`optimized_weight_blend_clipped` 提升到：

```text
0.11865
```

随后进一步比较不同分位数阈值，发现基于训练集 `SalePrice` 的 q993 上限效果最好，对应提交 `20260507_opt_clip_q993.csv` 的 Public Score 为：

```text
0.11805
```

因此，最终方案继续沿用 q993 高价硬裁剪。该操作不改变底层模型，但直接改变提交文件中的预测值，并被 Kaggle 反馈证明有效。

## 8.7 Target Encoding 增量

在高级特征和高价裁剪基础上，项目进一步引入 OOF Target Encoding。该方法主要用于补充类别变量中的价格水平信息，尤其是 `Neighborhood`、`MSSubClass`、`HouseStyle`、`SaleType` 以及若干组合类别。

为避免目标泄漏，训练集 Target Encoding 使用 OOF 方式生成。测试集则使用完整训练集统计，并通过平滑处理降低小样本类别的波动。

Target Encoding 独立方案中，`te_conservative_blend_clip_q993.csv` 的 Public Score 达到：

```text
0.11802
```

这一结果已经略优于上一轮 q993 裁剪方案的 `0.11805`。更重要的是，当 TE 保守融合结果与上一轮最好结果做 50/50 log 融合后，Public Score 进一步提升到 `0.11765`。

这说明 Target Encoding 提供了与原有高级特征体系不同的增量信息。它不是单独大幅替代原方案，而是作为互补信号参与最终融合。

## 8.8 最终融合方式

最终提交文件来自两个较强方案的 log 空间融合：

1. 上一轮最好结果：`20260507_opt_clip_q993.csv`。
2. Target Encoding 保守融合裁剪结果：`20260507_te_conservative_blend_clip_q993.csv`。

融合方式为：

```text
final = expm1(0.5 * log1p(pred_previous_best) + 0.5 * log1p(pred_te_conservative))
```

融合后继续保持 q993 高价上限。该方式有两个优点：

1. 融合在 log 价格尺度上进行，与 RMSLE 指标更一致。
2. 两个输入方案来自不同特征体系，具有一定互补性。

最终结果 `0.11765` 说明该融合确实带来了有效增量。

## 8.9 为什么选择该方案

选择最终方案的主要理由如下：

1. 它是当前所有已提交方案中 Kaggle Public Score 最低的方案。
2. 它保留了前几轮已经验证有效的高级特征工程和 q993 高价裁剪。
3. 它引入了 OOF Target Encoding，补充了类别变量的价格水平信息。
4. 它没有依赖外部数据或不可解释的泄漏信息，流程可复现。
5. 它的提升路径清楚：从 baseline 到高级融合，再到裁剪校准，最后由 TE 融合提供增量。

因此，该方案适合作为本项目当前阶段的最终方案。

## 8.10 本章小结

最终方案可以概括为“高级特征工程 + 多模型融合 + q993 高价裁剪 + OOF Target Encoding + 50/50 log 融合”。该方案在 Kaggle Public Leaderboard 上取得 `0.11765`，是当前项目中最好的提交结果。它既有较好的实际分数，也能从数据处理、特征构造和模型融合角度给出清晰解释。

## 8.11 本章引用材料

- `run_target_encoding_experiment.py`
- `house_price/advanced_preprocessing.py`
- `house_price/target_encoding.py`
- `experiments/experiment_log.csv`
- `reports/final_report/tables/experiment_results_summary.csv`
- `reports/modeling/20260507_target_encoding_experiment_report.md`
