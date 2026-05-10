# 3. 数据理解与探索性数据分析

## 3.1 分析目的

探索性数据分析的目标是理解数据结构、识别影响房价的关键因素，并为后续预处理、特征工程和建模策略提供依据。本项目的 EDA 不仅用于展示数据分布，还直接影响了后续建模决策，例如目标变量 log 变换、异常值删除、缺失值语义填充、质量类变量序数编码、高价尾部控制等。

本章主要围绕以下问题展开：

- `SalePrice` 是否存在偏态分布，是否需要变换？
- 哪些数值特征与房价关系最强？
- 类别变量是否具有明显价格分层？
- 缺失值是数据质量问题，还是业务含义的一部分？
- 训练集与测试集分布是否存在明显差异？
- 是否存在会干扰模型学习的异常样本？

## 3.2 目标变量分布

训练集目标变量 `SalePrice` 的原始分布明显右偏。大部分房屋价格集中在中低价区间，少数高价房形成长尾。统计上，原始 `SalePrice` 的偏度为 `1.8829`，最大值为 `755000`，显著高于中位数 `163000`。

![SalePrice 原始分布](../eda/figures/target_distribution.png)

对 `SalePrice` 进行 `log1p` 变换后，分布明显更接近对称，偏度下降到 `0.1213`。这说明 log 变换能够有效缓解高价长尾对模型训练的影响。

![log1p(SalePrice) 分布](../eda/figures/target_log_distribution.png)

这一观察直接决定了后续建模策略：训练阶段使用 `log1p(SalePrice)` 作为目标变量，预测后再通过 `expm1` 还原为原始价格。该处理方式也与 Kaggle 的 RMSLE 指标一致。

引用材料：

- `reports/final_report/tables/target_summary_for_report.csv`
- `reports/eda/tables/target_summary.csv`

## 3.3 缺失值结构

训练集和测试集都存在较多缺失单元格。训练集共有 7829 个缺失单元格，测试集共有 7878 个缺失单元格。缺失最多的字段包括 `PoolQC`、`MiscFeature`、`Alley`、`Fence`、`MasVnrType`、`FireplaceQu`、`LotFrontage` 等。

![训练集缺失值](../eda/figures/missing_train.png)

![测试集缺失值](../eda/figures/missing_test.png)

从字段含义看，许多高缺失字段并不是普通意义上的脏数据。例如：

- `PoolQC` 缺失通常表示房屋没有泳池。
- `Alley` 缺失通常表示没有巷道通道。
- `Fence` 缺失通常表示没有围栏。
- `FireplaceQu` 缺失通常表示没有壁炉。
- `Garage*` 系列字段缺失通常表示没有车库。
- `Bsmt*` 系列字段缺失通常表示没有地下室。

因此，后续预处理中不能简单删除缺失值，也不应对所有缺失值机械地填充均值。更合理的方式是结合业务含义，将“设施不存在”类缺失填充为 `None` 或 0，并为缺失字段保留缺失指示器。

引用材料：

- `reports/final_report/tables/missing_overview.csv`
- `reports/eda/tables/missing_train.csv`
- `reports/eda/tables/missing_test.csv`

## 3.4 数值特征与房价关系

相关性分析显示，`OverallQual` 是与房价相关性最高的单一数值特征，相关系数为 `0.7910`。此外，`GrLivArea`、`GarageCars`、`GarageArea`、`TotalBsmtSF`、`1stFlrSF` 等面积或容量相关变量也与房价高度相关。

![数值特征相关性条形图](../eda/figures/correlation_bar.png)

![关键数值特征相关性热力图](../eda/figures/correlation_heatmap.png)

这说明房价主要受以下几类因素影响：

- 房屋整体质量
- 居住面积和地下室面积
- 车库容量和车库面积
- 卫浴数量和房间数量
- 建造年份和翻新年份

这些观察支撑了后续构造面积组合特征、质量交互特征和年份差值特征的做法。

引用材料：

- `reports/eda/tables/numeric_correlations.csv`
- `reports/eda/figures/correlation_bar.png`
- `reports/eda/figures/correlation_heatmap.png`

## 3.5 面积变量分析

`GrLivArea` 表示地上居住面积，是房价预测中最关键的面积变量之一。散点图显示，`GrLivArea` 与 `SalePrice` 整体呈明显正相关关系，面积越大，房价通常越高。

![GrLivArea 与 SalePrice](../eda/figures/grlivarea_saleprice.png)

但图中也存在两个经典异常点：`Id=524` 和 `Id=1299`。这两个样本的 `GrLivArea` 极大，但销售价格相对偏低，会干扰模型学习“面积越大价格越高”的主趋势。因此，后续正式训练前删除了这两个训练样本。

`TotalBsmtSF` 与房价也呈正相关，但关系不如 `GrLivArea` 线性。地下室面积仍然具有价值，但更适合与其他面积变量组合，例如总面积、完成地下室面积比例等。

![TotalBsmtSF 与 SalePrice](../eda/figures/totalbsmtsf_saleprice.png)

## 3.6 质量变量分析

`OverallQual` 描述房屋整体材料和完成质量，是最强的单一特征。箱线图显示，随着 `OverallQual` 等级提高，房价中位数稳定上升，价格分布也整体上移。

![OverallQual 分组箱线图](../eda/figures/overallqual_boxplot.png)

质量类变量的建模启发是：这类变量具有天然顺序，适合进行序数编码。例如 `Ex > Gd > TA > Fa > Po`，如果完全 one-hot，可能损失等级顺序信息。因此后续对 `ExterQual`、`KitchenQual`、`BsmtQual`、`GarageQual` 等变量进行了有序数值映射。

引用材料：

- `reports/eda/tables/overallqual_price.csv`

## 3.7 类别变量与位置因素

类别变量中，`Neighborhood` 的价格分层最明显。不同社区的房价中位数差异较大，例如 `NridgHt`、`NoRidge`、`StoneBr` 等社区整体价格较高，而 `MeadowV`、`IDOTRR`、`BrDale` 等社区整体价格较低。

![Neighborhood 中位房价](../eda/figures/neighborhood_median_price.png)

这说明 `Neighborhood` 不是普通无信息类别，而是携带了强烈的位置溢价、社区档次和房屋结构差异。后续建模中既保留了 one-hot 编码，也在 Target Encoding 实验中重点使用了 `Neighborhood` 及其与质量、住宅类型的组合。

其他类别变量也有明显作用。例如，`ExterQual` 随质量等级提升呈现清晰价格分层；`HouseStyle` 不同类别之间也存在一定价格差异。

![ExterQual 与 SalePrice](../eda/figures/saleprice_by_exterqual.png)

![HouseStyle 与 SalePrice](../eda/figures/saleprice_by_house_style.png)

引用材料：

- `reports/eda/tables/neighborhood_price.csv`
- `reports/eda/tables/categorical_cardinality.csv`

## 3.8 年份变量分析

年份变量能够反映房屋新旧程度。`YearBuilt` 与房价整体正相关，新建房屋通常价格更高；`YearRemodAdd` 也呈类似趋势，说明翻新年份较新的房屋通常更有价值。

![YearBuilt 与 SalePrice](../eda/figures/yearbuilt_saleprice.png)

![YearRemodAdd 与 SalePrice](../eda/figures/yearremod_saleprice.png)

原始年份本身不是最直接的业务表达。后续特征工程中，将年份转换为 `HouseAge = YrSold - YearBuilt`、`RemodAge = YrSold - YearRemodAdd`，并构造是否翻新、是否新房等二值变量，使特征更接近业务含义。

## 3.9 训练集与测试集分布对比

训练集与测试集在核心变量上整体分布接近。例如 `GrLivArea`、`OverallQual` 的主要分布区域相似，这为模型泛化提供了基本前提。

![Train/Test GrLivArea 分布](../eda/figures/train_test_grlivarea.png)

![Train/Test OverallQual 分布](../eda/figures/train_test_overallqual.png)

但部分变量仍存在偏态或尾部差异，例如 `LotArea` 的右偏非常明显，因此需要对偏态数值变量进行变换，并在训练和测试上使用完全一致的预处理流程。

![Train/Test LotArea 分布](../eda/figures/train_test_lotarea.png)

引用材料：

- `reports/eda/tables/train_test_numeric_drift.csv`
- `reports/eda/tables/train_test_categorical_drift.csv`

## 3.10 EDA 对建模策略的启发

综合以上分析，EDA 对后续建模形成了以下直接指导：

1. `SalePrice` 右偏明显，应使用 `log1p(SalePrice)` 作为训练目标。
2. `GrLivArea` 极大但价格偏低的两个异常样本应从训练集中删除。
3. 缺失值需要按业务语义处理，不能简单删除。
4. 面积、质量、年份、设施存在性应作为特征工程重点。
5. 质量类变量具有顺序，应使用序数编码。
6. `Neighborhood` 等强类别变量应重点保留，并可尝试 Target Encoding。
7. 偏态数值变量需要 log 变换。
8. 高价尾部样本会显著影响提交结果，后续需要检查预测分布并控制高价外推风险。

这些结论贯穿了后续的数据预处理、特征工程、模型融合和高价尾部校准过程。

## 3.11 本章引用材料

- EDA 完整报告：`reports/eda/20260506_full_eda_report.md`
- EDA 图表导览：`reports/eda/20260506_eda_visual_guide.md`
- EDA 图表目录：`reports/eda/figures/`
- EDA 表格目录：`reports/eda/tables/`
