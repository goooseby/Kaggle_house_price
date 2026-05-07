# 20260507 新一轮模型方案设计

## 1. 当前状态

当前最好 Kaggle Public Score：

```text
20260507_opt_clip_q993.csv: 0.11805
```

从已有实验看，当前提升主要来自：

- 更强特征工程
- 多模型融合
- 高价尾部裁剪

但目前路线的问题也很清楚：

- 普通 CV 和 Public Score 不一致。
- stacking 本地分数偏乐观。
- 后处理裁剪已经进入小收益区间。
- 继续在 q993 附近调裁剪，可能只是在 Public Leaderboard 上过拟合。

因此下一轮目标应从“调提交文件”切换为：

> 用新评分面板指导更强模型和新特征体系，争取结构性提升。

## 2. 后续采用的新评分标准

旧标准：

```text
只看普通 CV RMSE
```

新标准：

```text
普通 CV + 高价尾部误差 + 预测分布风险
```

每个新模型必须输出：

| 指标 | 说明 |
| --- | --- |
| `cv_rmse` | 普通 OOF log RMSE |
| `tail_rmse_top_10pct` | 高价 top 10% OOF RMSE |
| `tail_rmse_top_5pct` | 高价 top 5% OOF RMSE |
| `tail_rmse_top_1pct` | 高价 top 1% OOF RMSE |
| `tail_bias_top_5pct` | 高价 top 5% 系统性偏差 |
| `pred_p99_minus_train_p99` | 预测 p99 相对训练集 p99 的差距 |
| `pred_max_over_train_q993` | 预测最大值相对训练集 q993 |
| `n_pred_above_train_q993` | 超过训练集 q993 的测试样本数 |
| `test_pred_tail_excess_sum_q993` | 测试预测超过 q993 的尾部超额总量 |

评分使用 `house_price/validation.py` 中的工具函数。

## 3. 为什么不能继续只调裁剪

本轮裁剪结果：

| 方案 | Public Score |
| --- | ---: |
| q997 | 0.11865 |
| q995 | 0.11818 |
| q993 | 0.11805 |
| q990 | 0.11841 |

这说明 q993 附近确实更优，但收益已经从上一轮的 `0.003` 降到本轮的 `0.0006`。

继续搜索 q992/q994 可能还有小提升，但很难带来大幅进步。

如果目标是进入更高排名段，下一步必须让模型本身更强。

## 4. 高价值模型方向

### 4.1 方案 A：OOF Target Encoding 特征体系

这是下一轮最高优先级。

当前 one-hot 对高基数类别的利用有限，尤其是：

- `Neighborhood`
- `Neighborhood + OverallQual`
- `MSSubClass`
- `Exterior1st`
- `Exterior2nd`
- `SaleType`
- `Condition1`
- `GarageType`

建议新增 OOF target encoding：

```text
category -> mean(log1p(SalePrice))
```

必须用 OOF 方式生成训练集编码，避免泄漏。

测试集使用全量训练集统计，并做平滑：

```text
encoded = (sum_y + global_mean * smoothing) / (count + smoothing)
```

预期收益：

- 提升线性模型对类别信息的表达能力。
- 帮助树模型捕捉位置和质量组合。
- 比继续加 one-hot 可能更有效。

风险：

- 如果不做 OOF 会泄漏。
- 稀有类别容易过拟合，需要平滑。

建议模型：

- Ridge / ElasticNet + target encoding
- XGBoost / LightGBM + target encoding
- target encoding 与 one-hot 特征并行比较

### 4.2 方案 B：CatBoost 原生类别模型

当前 CatBoost 使用的是 one-hot 后的数值矩阵，并没有发挥 CatBoost 原生处理类别特征的优势。

建议新建一条 CatBoost raw pipeline：

- 保留原始类别列
- 缺失类别填 `None`
- 数值列做合理缺失填充
- 类别列传给 CatBoost 的 `cat_features`
- 目标仍然是 `log1p(SalePrice)`

预期收益：

- CatBoost 能自动处理类别组合和 target statistics。
- 对 `Neighborhood`、`MSSubClass`、质量类字段可能更强。
- 可能提供和线性模型不同的预测视角，有利于融合。

风险：

- Public Score 未必单模型很强。
- 需要用新评分面板检查高价尾部。

### 4.3 方案 C：分层 CV 与价格分桶建模

现有普通 KFold 对高价区间不敏感。

下一轮训练应引入：

```text
StratifiedKFold on SalePrice bins
```

分桶方式：

```text
pd.qcut(log1p(SalePrice), q=10)
```

作用：

- 每折保留相近的高价样本比例。
- tail metrics 更稳定。
- 减少某一折高价房过少导致的误判。

这不是一个模型，而是所有新模型都应采用的验证基础。

### 4.4 方案 D：高价残差校准模型

现在硬裁剪有效，说明高价预测存在系统性问题。

比硬裁剪更高级的方式是训练 residual 校准模型。

流程：

1. 使用主模型产生 OOF 预测。
2. 计算 residual：

```text
residual = y_true_log - oof_pred_log
```

3. 只在高价区间或高预测区间训练校准模型。
4. 对测试集中高预测样本修正 log prediction。

候选校准特征：

- `Neighborhood`
- `OverallQual`
- `GrLivArea`
- `TotalBsmtSF`
- `LotArea`
- `YearBuilt`
- `GarageCars`
- 是否高端街区
- 是否极大面积但非高端街区

预期收益：

- 比全局硬裁剪更精细。
- 有机会保留真实豪宅，同时压低虚高预测。

风险：

- 数据很少，高价样本更少。
- 极易过拟合，需要非常保守。

建议作为第二批实验，不作为第一批。

### 4.5 方案 E：模型族融合，而不是单特征矩阵堆模型

目前多个模型多基于同一套特征矩阵，差异性不足。

下一轮应该构造不同“视角”的模型族：

| 模型族 | 特征体系 | 代表模型 |
| --- | --- | --- |
| Linear OneHot | one-hot + log 数值 + robust scaling | Ridge / ElasticNet / Lasso |
| Target Encoded | OOF target encoding + 数值特征 | Ridge / XGBoost / LightGBM |
| Raw CatBoost | 原始类别 + 数值填充 | CatBoost |
| Tree Numeric | ordinal + target encoding + 原始/轻变换数值 | XGBoost / LightGBM |
| High Price Calibrator | 主模型预测 + 高价特征 | Ridge / Huber / LightGBM |

然后融合时看模型族多样性，而不是简单把相似模型平均。

这可能比继续调单个 XGBoost/LightGBM 参数更有价值。

## 5. 可以更暴力尝试的方向

如果算力和时间不担心，可以加入更暴力的搜索。

### 5.1 Optuna 多目标调参

不要只优化 `cv_rmse`，而是优化组合目标：

```text
score = cv_rmse
      + 0.15 * tail_rmse_top_5pct
      + 0.05 * abs(tail_bias_top_5pct)
      + penalty(pred_max_over_train_q993)
```

这样调参会自然避开高价尾部外推严重的模型。

### 5.2 多随机种子 bagging

对强模型做多 seed：

- XGBoost seed 组
- LightGBM seed 组
- CatBoost seed 组

然后平均。

这通常能减少方差，但不会解决系统性偏差，所以必须配合新评分面板。

### 5.3 遗传/随机权重融合

基于 OOF 预测搜索融合权重，但必须加入约束：

- 权重非负
- 单模型权重上限
- 模型族权重上限
- tail risk penalty

不再允许当前 `ridge_stack` 那种负权重大幅摆动。

### 5.4 异常样本策略搜索

当前只删除了两个经典异常点。

可以系统测试：

- 是否删除更大面积异常点
- 是否保留高价豪宅
- 是否对 `Id=2550` 类似测试样本设计专门校准

注意：这类方向容易 LB 过拟合，应放在模型主体提升之后。

## 6. 下一轮建议实验包

我建议下一轮不要一次性把所有复杂方向都混在一起，而是做一个完整但可追溯的实验包。

### 实验包 1：Target Encoding + 新评分面板

产物：

- `house_price/target_encoding.py`
- `run_target_encoding_experiment.py`
- OOF 预测
- 测试集预测
- 新评分面板 CSV
- 候选提交 2-3 个

模型：

- Ridge TE
- ElasticNet TE
- XGBoost TE
- LightGBM TE

提交候选：

- 最佳 TE 融合
- 最佳 TE 融合 + q993 裁剪
- TE 与当前 best 的保守融合

### 实验包 2：Raw CatBoost

产物：

- `run_catboost_raw_experiment.py`
- CatBoost 原生类别模型
- 多 seed bagging
- 新评分面板

提交候选：

- raw CatBoost
- raw CatBoost + q993 裁剪
- raw CatBoost 与当前 best 融合

### 实验包 3：模型族融合

输入：

- 当前 optimized/simple
- TE 模型预测
- raw CatBoost 预测

融合：

- 非负权重
- 模型族权重约束
- tail risk penalty

提交候选：

- family blend
- family blend + q993
- family blend + adaptive high-price cap

## 7. 推荐执行顺序

下一步优先做：

```text
Target Encoding + 新评分面板
```

原因：

- 和当前特征体系差异大。
- 可能带来结构性提升。
- 对 `Neighborhood` 这类强类别变量更友好。
- 可同时服务线性模型和树模型。

第二步：

```text
Raw CatBoost 原生类别模型
```

原因：

- 与 one-hot / target encoding 都不同。
- 可能提供融合多样性。

第三步：

```text
模型族融合
```

原因：

- 等有了新模型族预测后再融合更有意义。

## 8. 本轮设计结论

后续不是放弃高价校准，而是把它从“提交后处理”提升为“模型评估标准”。

新模型必须同时满足：

1. 普通 CV 有竞争力。
2. 高价 tail RMSE 不恶化。
3. 高价 tail bias 不明显正偏。
4. 测试预测分布不过度外推。
5. 与已有模型有足够差异性。

下一轮最值得投入的是：

> OOF Target Encoding 特征体系 + 新评分面板。

如果这个方向有效，再把 raw CatBoost 和模型族融合接进来，才有机会向更高排名段推进。
