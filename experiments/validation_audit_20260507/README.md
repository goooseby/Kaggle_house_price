# 20260507 验证审计说明

本目录记录“本地评分如何更接近 Kaggle Public Score”的审计结论。

相关产物：

- `experiments/submission_score_audit.csv`
- `experiments/validation_metric_correlations.csv`
- `experiments/score_prediction_comparison.csv`
- `reports/20260507_validation_audit_report.md`
- `house_price/validation.py`

## 结论

旧方法主要看普通 CV RMSE；这个指标对 Public Score 的排序解释能力不足。

当前更可靠的评分面板必须包含：

- 普通 CV RMSE
- 高价 top 10% / 5% / 1% OOF RMSE
- 高价 top 10% / 5% / 1% OOF bias
- 测试集预测 p99
- 测试集预测 max / train q993
- 测试集 tail excess sum q993
- 测试集 tail excess sum q997

后续任何新模型，不能只用普通 CV 决定是否提交。

## 使用原则

新模型筛选顺序：

1. 先看普通 CV 是否有竞争力。
2. 再看高价 tail RMSE 和 tail bias 是否健康。
3. 再看测试集预测分布是否有高价外推风险。
4. 最后结合少量 Kaggle 提交反馈验证方向。

如果普通 CV 提升但高价尾部风险明显恶化，不作为优先提交候选。
