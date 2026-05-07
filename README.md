# Kaggle House Price

这是 Kaggle 入门题 **House Prices - Advanced Regression Techniques** 的本地项目。

当前状态：

- 第一版 baseline 已跑通
- 已生成 Kaggle 提交文件并完成一次提交
- Kaggle Public Score: `0.12859`
- 排名约前 `30%`
- 下一步准备进入完整优化回合

## 快速运行

```powershell
python main.py
```

这会重新生成：

- `outputs/eda_report.md`
- `outputs/cv_results.csv`
- `outputs/submission_baseline.csv`

## 目录说明

```text
house_price/     核心代码
docs/            项目文档和优化计划
experiments/     实验记录
reports/         人工整理后的实验报告
submissions/     Kaggle 候选提交文件
outputs/         自动生成产物，不提交 Git
```

## 当前重要文档

- [项目文档索引](docs/README.md)
- [完整优化计划](docs/06_full_optimization_plan.md)
- [环境安装说明](docs/07_environment_setup.md)
- [实验记录](experiments/experiment_log.csv)

## 下一步

安装优化依赖后，开始完整优化回合：

```powershell
pip install -r requirements-optimization.txt
```

然后我们会同时推进：

- 更可靠的 OOF 验证
- 更完整的特征工程
- XGBoost / LightGBM / CatBoost 等强模型
- 系统调参
- 融合与预测裁剪
- 候选提交文件和实验报告
