# 2. 数据集说明

## 2.1 数据来源与文件组成

本项目使用 Kaggle House Prices 竞赛提供的数据文件。原始数据目录中包含：

- `train.csv`：训练集，包含房屋属性和目标变量 `SalePrice`。
- `test.csv`：测试集，包含房屋属性，不包含 `SalePrice`。
- `sample_submission.csv`：Kaggle 要求的提交文件格式示例。
- `data_description.txt`：字段说明文档，解释各个变量的业务含义。

在项目代码中，数据通过 `house_price.data.load_raw_data` 统一读取。训练集和测试集后续会经过相同的预处理与特征工程流程，以保证训练和预测阶段的特征空间一致。

## 2.2 数据规模

根据报告材料生成脚本统计，原始数据规模如下：

| 数据集 | 行数 | 列数 | 特征列数 | 是否包含 SalePrice | 缺失单元格 | 重复行 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| train | 1460 | 81 | 80 | yes | 7829 | 0 |
| test | 1459 | 80 | 80 | no | 7878 | 0 |
| sample_submission | 1459 | 2 | 0 | yes | 0 | 0 |

训练集包含 1460 条样本，测试集包含 1459 条样本。训练集比测试集多一列 `SalePrice`，其余 80 个字段为房屋属性。数据集中没有重复行。

本节表格来源：

- `reports/final_report/tables/dataset_profile.csv`

## 2.3 特征类型

从数据类型看，训练集中除目标变量外共有 80 个原始特征。其中：

| 数据类型 | 特征数量 |
| --- | ---: |
| float64 | 3 |
| int64 | 34 |
| str | 43 |

可以看到，类别型变量数量较多，这也是本任务的一个重要特点。房价不仅由面积、年份等连续数值决定，也受到社区、建筑类型、质量等级、交易条件等类别信息影响。

本节表格来源：

- `reports/final_report/tables/feature_type_summary.csv`

## 2.4 特征业务分组

根据字段含义，可以将原始特征划分为以下几类：

| 特征组 | 特征数量 | 示例特征 |
| --- | ---: | --- |
| 房屋类型与 zoning | 4 | `MSSubClass`, `MSZoning`, `BldgType`, `HouseStyle` |
| 地块与位置 | 12 | `LotFrontage`, `LotArea`, `Street`, `Alley`, `Neighborhood`, `Condition1` |
| 建筑质量与材料 | 11 | `OverallQual`, `OverallCond`, `Exterior1st`, `Exterior2nd`, `ExterQual`, `Foundation` |
| 地下室 | 9 | `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinSF1`, `TotalBsmtSF` |
| 室内面积与功能 | 19 | `GrLivArea`, `1stFlrSF`, `2ndFlrSF`, `FullBath`, `KitchenQual`, `Fireplaces` |
| 车库 | 7 | `GarageType`, `GarageYrBlt`, `GarageFinish`, `GarageCars`, `GarageArea` |
| 外部设施 | 11 | `WoodDeckSF`, `OpenPorchSF`, `PoolArea`, `PoolQC`, `Fence`, `MiscFeature` |
| 时间与交易 | 6 | `YearBuilt`, `YearRemodAdd`, `MoSold`, `YrSold`, `SaleType`, `SaleCondition` |

这种分组有助于后续解释特征工程。例如，面积类变量适合构造总面积，质量类变量适合构造综合质量，年份类变量适合转换为房龄和翻新时间，设施类变量则常常需要结合缺失值含义构造“是否存在该设施”的二值特征。

本节表格来源：

- `reports/final_report/tables/feature_domain_groups.csv`

## 2.5 目标变量 SalePrice 概况

训练集目标变量 `SalePrice` 的统计情况如下：

| 变量 | 均值 | 标准差 | 最小值 | 中位数 | 95% 分位数 | 最大值 | 偏度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SalePrice | 180921.20 | 79442.50 | 34900.00 | 163000.00 | 326100.00 | 755000.00 | 1.8829 |
| log1p(SalePrice) | 12.0241 | 0.3994 | 10.4603 | 12.0015 | 12.6950 | 13.5345 | 0.1213 |

原始 `SalePrice` 呈现明显右偏分布，最大值达到 755000，远高于中位数 163000。对目标变量进行 `log1p` 变换后，偏度从 1.8829 降到 0.1213，分布更接近对称。这一结果支持后续建模中使用 log 目标变量。

本节表格来源：

- `reports/final_report/tables/target_summary_for_report.csv`
- `reports/eda/figures/target_distribution.png`
- `reports/eda/figures/target_log_distribution.png`

## 2.6 缺失值概况

原始数据中存在较多缺失值，但这些缺失值并不都表示数据质量问题。很多字段的缺失具有明确业务含义，例如：

- `Alley` 缺失通常表示没有巷道通道。
- `PoolQC` 缺失通常表示没有泳池。
- `Fence` 缺失通常表示没有围栏。
- 地下室相关字段缺失通常表示没有地下室。
- 车库相关字段缺失通常表示没有车库。

整体缺失情况如下：

| 数据集 | 存在缺失的特征数 | 缺失单元格数 | 缺失单元格比例 | 缺失最多的字段示例 |
| --- | ---: | ---: | ---: | --- |
| train | 19 | 7829 | 6.6202% | `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, `LotFrontage`, `GarageType` |
| test | 33 | 7878 | 6.7489% | `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`, `LotFrontage`, `GarageYrBlt` |

因此，本项目后续并不是简单删除缺失值，而是结合字段含义进行分组处理。对于“设施不存在”类缺失，通常填充为 `None` 或 0；对于少数真正未知的数值缺失，再使用中位数或其他稳健策略进行填充。

本节表格来源：

- `reports/final_report/tables/missing_overview.csv`
- `reports/eda/tables/missing_train.csv`
- `reports/eda/tables/missing_test.csv`
- `reports/eda/figures/missing_train.png`
- `reports/eda/figures/missing_test.png`

## 2.7 数据集特点总结

综合来看，本数据集具有以下特点：

1. 样本量较小，训练集只有 1460 条记录，因此模型验证容易受到样本划分影响。
2. 特征数量较多，且类别型变量占比较高，需要认真处理类别编码。
3. 目标变量价格分布右偏明显，需要进行 log 变换。
4. 缺失值数量较多，但很多缺失具有业务含义，不能简单视为异常。
5. 房价受多类因素共同影响，包括位置、面积、质量、年份、设施和交易条件。
6. 高价房形成长尾，后续实验表明高价尾部预测会明显影响 Kaggle 提交分数。

这些特点决定了本项目不能只依赖单一模型或简单预处理，而需要结合 EDA、稳健预处理、特征工程、多模型融合和预测分布检查，逐步提高模型效果。

## 2.8 本章补充材料

为了支撑本章写作，本轮新增了报告材料生成脚本：

```powershell
conda run -n kaggle_house python scripts/generate_final_report_tables.py
```

该脚本生成的表格位于：

```text
reports/final_report/tables/
```

包含：

- `dataset_profile.csv`
- `feature_type_summary.csv`
- `feature_domain_groups.csv`
- `missing_overview.csv`
- `target_summary_for_report.csv`
