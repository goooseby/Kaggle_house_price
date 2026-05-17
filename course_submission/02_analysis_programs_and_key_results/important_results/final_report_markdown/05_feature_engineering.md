# 5. 特征工程

## 5.1 特征工程目标

特征工程的目标是在原始字段和基础预处理结果之上，构造更贴近房屋价格形成机制的变量。房价不是由单一因素决定，而是由面积、质量、位置、年份、设施条件和交易状态共同影响。因此，本项目的特征工程主要围绕以下方向展开：

- 将分散的面积信息整合为更完整的规模指标。
- 将年份字段转换为更有解释性的房龄变量。
- 将设施是否存在显式表达为二值特征。
- 构造比例特征，描述房屋结构。
- 构造质量与面积、质量与位置之间的交互。
- 在后期实验中引入 OOF Target Encoding，提取类别变量的价格水平信息。

## 5.2 面积类组合特征

EDA 显示，`GrLivArea`、`TotalBsmtSF`、`1stFlrSF` 等面积变量与房价高度相关。原始数据中的面积信息分散在多个字段中，因此构造了多个组合面积特征：

```text
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalLivingSF = GrLivArea + TotalBsmtSF
TotalFinishedSF = BsmtFinSF1 + BsmtFinSF2 + 1stFlrSF + 2ndFlrSF
TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
```

这些特征比单一面积字段更全面地描述房屋规模。尤其是 `TotalSF` 和 `TotalLivingSF`，能够同时考虑地上居住面积和地下室空间，对房价预测具有较强解释力。

## 5.3 卫浴与功能特征

卫浴数量反映房屋居住便利性。原始数据中完整浴室和半浴室分布在多个字段中，因此构造综合卫浴数：

```text
TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
```

该特征将地上和地下室卫浴信息合并，同时用 0.5 表示半浴室的相对价值，使其比单独使用多个浴室字段更简洁。

## 5.4 年份与房龄特征

原始年份字段本身有信息，但直接使用年份不如使用“年龄”更符合业务含义。项目中构造了：

```text
HouseAge = YrSold - YearBuilt
RemodAge = YrSold - YearRemodAdd
GarageAge = YrSold - GarageYrBlt
```

同时构造：

- `IsRemodeled`：是否翻新过。
- `IsNewHouse`：是否销售年份等于建造年份。

这些变量能够表达房屋新旧程度和翻新状态。对于房价预测来说，新房、近期翻新房通常具有更高价格水平。

## 5.5 设施存在性特征

缺失值分析表明，许多字段缺失意味着设施不存在。因此，项目中显式构造了设施存在性变量：

- `HasGarage`
- `HasBsmt`
- `HasFireplace`
- `HasPool`
- `Has2ndFloor`
- `HasPorch`

这类特征把“是否具有某项设施”直接交给模型学习。例如，有无车库、有无地下室、有无壁炉都可能影响房屋价值。

## 5.6 比例特征

绝对面积能够描述房屋规模，但比例特征可以描述结构。项目中构造了：

- `FinishedBasementRatio`
- `GarageAreaPerCar`
- `SecondFloorRatio`
- `PorchRatio`
- `LotAreaPerLivingSF`

例如，`FinishedBasementRatio` 表示地下室完成比例；`GarageAreaPerCar` 表示每个车位对应的车库面积；`LotAreaPerLivingSF` 表示地块面积与居住面积的比例。这些特征能够补充原始面积变量无法表达的结构信息。

## 5.7 质量交互特征

EDA 显示，`OverallQual` 是与房价关系最强的单一特征。质量对面积、车库、卫浴和房龄等因素具有放大或调节作用。例如，同样面积的房屋，质量更高通常价格更高；同样质量的房屋，社区和房龄不同也会造成价格差异。

因此，项目构造了以下交互特征：

- `OverallQual_TotalSF`
- `OverallQual_GrLivArea`
- `OverallQual_GarageCars`
- `OverallQual_TotalBathrooms`
- `OverallQual_YearBuilt`
- `Age_OverallQual`
- `Neighborhood_OverallQual`

这些交互特征用于增强模型对非线性关系的表达能力，尤其对线性模型有帮助。

## 5.8 Target Encoding 特征

在后期模型重设计阶段，本项目引入了 OOF Target Encoding 特征。Target Encoding 的基本思想是用类别分组的目标均值表示类别，例如不同 `Neighborhood` 的平均 log 房价。

为避免目标泄漏，训练集使用 OOF 方式生成编码：

1. 将训练集划分为多个 fold。
2. 对每个验证 fold，只使用其余训练 fold 计算类别目标均值。
3. 使用该统计量编码当前验证 fold。
4. 测试集使用完整训练集统计，并进行平滑处理。

本项目生成了 20 个 TE 特征，包括：

- 单字段 TE：`Neighborhood`、`MSSubClass`、`HouseStyle`、`SaleType`、`SaleCondition` 等。
- 组合字段 TE：`Neighborhood + OverallQual`、`Neighborhood + MSSubClass`、`OverallQual + KitchenQual` 等。

实验结果显示，TE 单独方案提升有限，但与上一轮最好方案进行 50/50 log 融合后取得了当前最好 Public Score `0.11758`。这说明 TE 特征提供了与原有模型不同的误差信息，具有一定增量价值。

对应代码与报告：

- `house_price/target_encoding.py`
- `reports/modeling/20260507_target_encoding_experiment_report.md`

## 5.9 特征工程结果

正式建模时，基础高级预处理矩阵包含 447 个特征；在 Target Encoding 实验中，额外加入 20 个 TE 特征。特征工程使模型不再只依赖原始字段，而是能够同时利用房屋规模、质量、结构、位置、年份、设施和类别价格水平等信息。

从实验结果看，特征工程对模型提升有明显贡献：baseline Public Score 为 `0.12859`，而在高级特征和多模型融合后，成绩提升到 `0.11865`；进一步加入高价校准和 TE 融合后，最终达到 `0.11758`。

## 5.10 本章小结

本项目的特征工程围绕房价形成逻辑展开。面积组合特征描述房屋规模，年份特征描述新旧程度，设施存在性特征表达功能条件，比例特征刻画结构差异，质量交互特征增强非线性表达，Target Encoding 则补充类别变量中的价格水平信息。这些特征共同构成后续模型训练和融合的基础。

## 5.11 本章引用材料

- `house_price/advanced_preprocessing.py`
- `house_price/target_encoding.py`
- `reports/final_report/tables/preprocessing_feature_summary.csv`
- `reports/final_report/tables/skew_transformed_columns.csv`
- `reports/modeling/20260507_target_encoding_experiment_report.md`
