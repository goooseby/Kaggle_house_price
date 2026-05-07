# House Prices EDA 图表导览

这份文档用于快速浏览本次 EDA 产出的图表和统计表。

完整分析报告见：

- [20260506_full_eda_report.md](20260506_full_eda_report.md)

统计表目录：

- [tables/](tables/)

图表目录：

- [figures/](figures/)

## 1. 数据整体情况

相关表格：

- [dataset_overview.csv](tables/dataset_overview.csv)：训练集、测试集、提交样例的行列数、缺失单元格数量、重复行数量
- [dtype_summary.csv](tables/dtype_summary.csv)：字段类型统计

用途：

这两个表先回答“数据有多大、有哪些类型、是否有重复行、缺失规模如何”。

## 2. 目标值 SalePrice 分布

相关表格：

- [target_summary.csv](tables/target_summary.csv)：`SalePrice` 和 `log1p(SalePrice)` 的描述统计

### 原始 SalePrice 分布

<img src="figures/target_distribution.png" alt="SalePrice Distribution" style="zoom: 50%;" />

说明：

这张图展示房价原始分布。可以看到明显右偏：大部分房子集中在中低价区间，少数高价房拉长右尾。

建模含义：

直接预测原始房价会让模型更容易受到高价样本影响。

### log1p(SalePrice) 分布

<img src="figures/target_log_distribution.png" alt="log1p SalePrice Distribution" style="zoom:50%;" />

说明：

这张图展示 `log1p(SalePrice)` 的分布。相比原始房价，它更接近对称分布。

建模含义：

这支持我们在模型中使用 `log1p(SalePrice)` 作为目标值，预测后再用 `expm1` 还原价格。

## 3. 缺失值结构

相关表格：

- [missing_train.csv](tables/missing_train.csv)：训练集缺失值统计
- [missing_test.csv](tables/missing_test.csv)：测试集缺失值统计

### 训练集缺失值

<img src="figures/missing_train.png" alt="Train Missing Values" style="zoom:50%;" />

说明：

这张图展示训练集中缺失比例最高的字段。`PoolQC`、`MiscFeature`、`Alley`、`Fence` 缺失比例很高。

建模含义：

这些缺失值大多不是脏数据，而是表示“没有泳池、没有杂项设施、没有巷道、没有围栏”。因此应该填成 `None`，而不是均值或直接删除。

### 测试集缺失值

<img src="figures/missing_test.png" alt="Test Missing Values" style="zoom:50%;" />

说明：

测试集的缺失模式和训练集类似，但也有少量字段在测试集出现额外缺失。

建模含义：

预处理必须同时考虑训练集和测试集。否则模型训练能通过，预测测试集时可能因为缺失或列不一致出错。

## 4. 数值特征相关性

相关表格：

- [numeric_correlations.csv](tables/numeric_correlations.csv)：数值特征与 `SalePrice` 的相关系数
- [numeric_skewness.csv](tables/numeric_skewness.csv)：数值特征偏度

### 相关性条形图

<img src="figures/correlation_bar.png" alt="Correlation Bar" style="zoom:50%;" />

说明：

这张图展示和 `SalePrice` 相关性最高的数值特征。

主要发现：

- `OverallQual` 是最强特征。
- `GrLivArea`、`GarageCars`、`GarageArea`、`TotalBsmtSF`、`1stFlrSF` 都很重要。
- 面积、质量、车库、地下室是价格预测的核心信息来源。

### 关键数值特征相关性热力图

<img src="figures/correlation_heatmap.png" alt="Correlation Heatmap" style="zoom:50%;" />

说明：

这张图展示核心数值特征之间的相关关系。

建模含义：

很多面积相关特征彼此相关，例如 `GarageCars` 和 `GarageArea`、`TotalBsmtSF` 和 `1stFlrSF`。线性模型需要正则化来处理这种共线性。

## 5. 面积与价格关系

### GrLivArea vs SalePrice

<img src="figures/grlivarea_saleprice.png" alt="GrLivArea vs SalePrice" style="zoom:50%;" />

说明：

`GrLivArea` 是地上居住面积。整体趋势是面积越大，价格越高。

重要异常点：

图中能看到少数面积很大但价格很低的点，特别是经典异常点 `Id=524` 和 `Id=1299`。

建模含义：

删除这两个异常点是合理的，因为它们会破坏面积和价格之间的主趋势。

### TotalBsmtSF vs SalePrice

<img src="figures/totalbsmtsf_saleprice.png" alt="TotalBsmtSF vs SalePrice" style="zoom:50%;" />

说明：

`TotalBsmtSF` 是地下室总面积。地下室越大，价格整体越高，但关系不如 `GrLivArea` 那么线性。

建模含义：

地下室面积有价值，但更适合和其他特征组合，例如总面积、地下室完成比例。

## 6. 房屋质量与价格

相关表格：

- [overallqual_price.csv](tables/overallqual_price.csv)：不同 `OverallQual` 下的价格统计

### OverallQual 分组箱线图

<img src="figures/overallqual_boxplot.png" alt="OverallQual Boxplot" style="zoom:50%;" />

说明：

这张图展示不同整体质量评分下的房价分布。

主要发现：

`OverallQual` 越高，房价中位数越高，且趋势非常稳定。

建模含义：

`OverallQual` 是本题最关键特征之一。后续构造 `OverallQual * TotalSF`、`OverallQual * GrLivArea` 等交互特征是有依据的。

## 7. 街区与价格

相关表格：

- [neighborhood_price.csv](tables/neighborhood_price.csv)：各街区价格统计

### Neighborhood 中位房价

<img src="figures/neighborhood_median_price.png" alt="Neighborhood Median Price" style="zoom:50%;" />

说明：

这张图展示不同街区的房价中位数差异。

主要发现：

街区之间价格差异很大。`NridgHt`、`NoRidge`、`StoneBr` 等街区明显更贵，`MeadowV`、`IDOTRR`、`BrDale` 等街区较低。

建模含义：

`Neighborhood` 是强类别特征，不能忽略。它不仅表示位置，还隐含社区等级、房屋档次等信息。

## 8. 类别特征与价格

相关表格：

- [categorical_cardinality.csv](tables/categorical_cardinality.csv)：类别特征取值数量和主导取值比例

### HouseStyle 与 SalePrice

<img src="figures/saleprice_by_house_style.png" alt="SalePrice by HouseStyle" style="zoom:50%;" />

说明：

不同房屋风格的价格分布存在差异，但也有较大重叠。

建模含义：

`HouseStyle` 有一定信息量，适合 one-hot 编码。

### ExterQual 与 SalePrice

<img src="figures/saleprice_by_exterqual.png" alt="SalePrice by ExterQual" style="zoom:50%;" />

说明：

外部材料质量越高，房价越高，趋势明显。

建模含义：

`ExterQual` 这类质量等级字段适合做序数编码，例如 `Ex > Gd > TA > Fa > Po`。

## 9. 年份变量

### YearBuilt vs SalePrice

<img src="figures/yearbuilt_saleprice.png" alt="YearBuilt vs SalePrice" style="zoom:50%;" />

说明：

建造年份越新，价格整体越高。

建模含义：

可以构造 `HouseAge = YrSold - YearBuilt`。

### YearRemodAdd vs SalePrice

<img src="figures/yearremod_saleprice.png" alt="YearRemodAdd vs SalePrice" style="zoom:50%;" />

说明：

翻新年份越新，价格整体也越高。

建模含义：

可以构造 `RemodAge = YrSold - YearRemodAdd`，以及是否翻新过的标记。

## 10. 训练集与测试集分布差异

相关表格：

- [train_test_numeric_drift.csv](tables/train_test_numeric_drift.csv)：数值特征 train/test 分布差异，按 KS statistic 排序
- [train_test_categorical_drift.csv](tables/train_test_categorical_drift.csv)：类别特征 train/test 取值差异

### LotArea 分布对比

<img src="figures/train_test_lotarea.png" alt="Train/Test LotArea" style="zoom:50%;" />

说明：

这张图比较训练集和测试集的 `LotArea` 分布。由于 `LotArea` 右偏明显，图中使用的是 log 后尺度。

建模含义：

`LotArea` 适合做 `log1p` 变换。

### GrLivArea 分布对比

<img src="figures/train_test_grlivarea.png" alt="Train/Test GrLivArea" style="zoom:50%;" />

说明：

训练集和测试集的 `GrLivArea` 主体分布比较接近，但高面积区域仍要小心。

建模含义：

高面积样本容易影响高价预测，也是后续 clipped 策略有效的原因之一。

### OverallQual 分布对比

<img src="figures/train_test_overallqual.png" alt="Train/Test OverallQual" style="zoom:50%;" />

说明：

训练集和测试集在 `OverallQual` 上整体接近。

建模含义：

核心质量特征没有明显 train/test 分布断裂，这对泛化是好信号。

## 11. 异常点和低方差特征

相关表格：

- [outlier_candidates.csv](tables/outlier_candidates.csv)：候选异常点
- [low_variance_features.csv](tables/low_variance_features.csv)：低方差特征

说明：

异常点表里最重要的是两个大面积低价样本：

- `Id=524`
- `Id=1299`

低方差特征中，`Utilities`、`Street`、`PoolArea`、`PoolQC` 等字段绝大多数样本取同一个值。

建模含义：

- 两个经典面积异常点应删除。
- 低方差特征单独贡献有限，但不一定必须删除，因为它们可能在少数特殊样本上有信息。

## 12. 总结

这组 EDA 图表支持我们当前建模方向：

1. 对 `SalePrice` 做 log 变换。
2. 对偏态数值特征做 `log1p`。
3. 对质量类字段做序数编码。
4. 对 `Neighborhood` 等强类别字段做 one-hot。
5. 删除 `GrLivArea` 极大但价格异常低的两个样本。
6. 构造面积、质量、年份、比例和交互特征。
7. 关注高价区间外推风险，这也解释了为什么 clipped 提交文件表现更好。
