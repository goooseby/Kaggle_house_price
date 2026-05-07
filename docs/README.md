# Kaggle House Prices Project Notes

这个目录用于说明我们在 Kaggle 入门题 **House Prices - Advanced Regression Techniques** 中做了什么、为什么这么做，以及下一步可以怎样继续优化。

当前项目已经有一版可以完整运行的 baseline：

- 读取原始数据
- 生成探索性分析报告
- 删除典型异常点
- 处理缺失值
- 构造基础特征
- 对类别特征做编码
- 使用多个回归模型做 5 折交叉验证
- 融合模型预测测试集
- 生成 Kaggle 可提交的 `submission_baseline.csv`

## 文档索引

- [01_problem_overview.md](01_problem_overview.md)：题目背景、预测目标、评价指标
- [02_data_understanding.md](02_data_understanding.md)：数据规模、字段类型、缺失值和关键特征
- [03_baseline_pipeline.md](03_baseline_pipeline.md)：当前 baseline 的完整流程
- [04_results_report.md](04_results_report.md)：当前模型表现、输出文件和结果解释
- [05_next_steps.md](05_next_steps.md)：后续优化路线
- [06_full_optimization_plan.md](06_full_optimization_plan.md)：下一轮完整优化计划
- [07_environment_setup.md](07_environment_setup.md)：下一轮优化所需环境

## 当前入口

运行项目：

```powershell
python main.py
```

主要输出：

- `outputs/eda_report.md`
- `outputs/cv_results.csv`
- `outputs/submission_baseline.csv`

## 当前阶段

当前阶段不是追求最终最高分，而是先建立一个可靠、可复现、可解释的建模流程。这个流程跑通后，后面调参、换模型、加特征才有稳定的比较基准。

## 当前提交反馈

第一版 baseline 已提交 Kaggle：

- Public Score: `0.12859`
- 排名约前 `30%`

下一步不再做小修小补，而是进行一个完整优化回合：验证体系、特征工程、强模型、调参、融合和候选提交一起推进。
