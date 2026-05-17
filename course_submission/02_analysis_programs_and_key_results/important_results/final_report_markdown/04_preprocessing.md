# 4. 数据预处理

## 4.1 预处理目标

数据预处理的目标是将原始房屋属性数据转换为稳定、无缺失、可编码、可用于模型训练的基础数据矩阵。由于本数据集同时包含数值变量、类别变量、等级变量、缺失值、偏态分布和异常样本，预处理不能只做机械填充，而应结合字段含义和模型需求进行处理。

本项目后期正式建模主要使用 `house_price.advanced_preprocessing.build_advanced_feature_matrix`。该流程遵循以下原则：

- 训练集和测试集合并处理，保证编码后特征列一致。
- 缺失值按业务含义分组填充。
- 对原始缺失信息构造缺失指示器。
- 对类别变量区分“有序类别”和“无序类别”。
- 对偏态数值变量进行变换。
- 对最终数值矩阵进行稳健缩放。

## 4.2 异常值处理

在 EDA 中，`Id=524` 和 `Id=1299` 被识别为典型的大面积低价异常样本。这两个样本的 `GrLivArea` 极大，但 `SalePrice` 明显偏低，会削弱模型对“居住面积越大，价格通常越高”这一主趋势的学习。

因此，正式建模前删除满足以下条件的训练样本：

```text
GrLivArea > 4000 且 SalePrice < 300000
```

该处理只应用于训练集，不应用于测试集。原因是测试集没有真实价格，无法判断某个样本是否属于“低价异常”；同时 Kaggle 要求对测试集全部样本给出预测。

处理后，训练样本数由 1460 减少为 1458。

对应代码：

- `house_price/data.py`

## 4.3 目标变量变换

原始 `SalePrice` 呈明显右偏分布，高价房形成长尾。为适配 Kaggle 的 RMSLE 指标，并降低高价样本对训练过程的影响，本项目将目标变量变换为：

```text
y = log1p(SalePrice)
```

模型训练、交叉验证和融合权重学习均在 log 价格尺度上进行。生成提交文件时，再通过：

```text
SalePrice = expm1(y_pred)
```

将预测结果还原为原始价格。

这一处理具有三点作用：

1. 与 RMSLE 指标的计算逻辑一致。
2. 缓解目标变量右偏问题。
3. 使模型更关注相对误差，而不是被少数高价样本的绝对误差主导。

## 4.4 缺失值语义处理

本数据集中的缺失值具有明显业务语义。许多缺失并不表示数据记录错误，而是表示某项设施不存在。因此，本项目按字段含义分组处理缺失值。

### 4.4.1 设施不存在类缺失

部分类别字段缺失表示房屋没有对应设施，例如：

- `PoolQC`：无泳池。
- `MiscFeature`：无杂项设施。
- `Alley`：无巷道通道。
- `Fence`：无围栏。
- `FireplaceQu`：无壁炉。
- `GarageType`、`GarageFinish`、`GarageQual`、`GarageCond`：无车库。
- `BsmtQual`、`BsmtCond`、`BsmtExposure`、`BsmtFinType1`、`BsmtFinType2`：无地下室。

这类字段填充为 `None`，使模型能够学习“设施不存在”本身对价格的影响。

### 4.4.2 面积、数量和年份类缺失

对于与设施不存在对应的数值字段，缺失值填充为 0，例如：

- `GarageYrBlt`
- `GarageArea`
- `GarageCars`
- `BsmtFinSF1`
- `BsmtFinSF2`
- `BsmtUnfSF`
- `TotalBsmtSF`
- `BsmtFullBath`
- `BsmtHalfBath`
- `MasVnrArea`

该处理的业务含义是：如果房屋没有对应设施，则其面积、数量或年份贡献为 0。

### 4.4.3 `LotFrontage` 分组填充

`LotFrontage` 表示临街长度，缺失较多，且与社区位置相关。不同社区的道路条件和地块形态可能不同，因此本项目按 `Neighborhood` 分组，用同社区的中位数填充：

```text
LotFrontage = median(LotFrontage | Neighborhood)
```

这种方式比全局中位数更符合业务语境。

### 4.4.4 少量普通缺失

对于少量类别字段，例如 `MSZoning`、`Electrical`、`KitchenQual`、`Exterior1st`、`Exterior2nd`、`SaleType`、`Utilities`，使用众数填充。对于 `Functional`，缺失值填充为 `Typ`，表示典型功能状态。

在前述规则之后，如仍有数值缺失，则使用中位数填充；如仍有类别缺失，则使用众数或 `None` 填充。

对应代码：

- `house_price/preprocessing.py`
- `house_price/advanced_preprocessing.py`

## 4.5 缺失指示器

在填充缺失值前，高级预处理流程会为所有存在缺失的字段构造缺失指示器：

```text
Column_WasMissing
```

例如，若 `LotFrontage` 原始值缺失，则新增：

```text
LotFrontage_WasMissing = 1
```

这样即使后续完成了数值填充，模型仍能识别该字段是否原本缺失。对于本数据集来说，缺失本身往往含有业务信息，因此缺失指示器是有意义的补充特征。正式建模矩阵中共生成 34 个缺失指示器。

引用材料：

- `reports/final_report/tables/preprocessing_feature_summary.csv`

## 4.6 类别变量基础处理

### 4.6.1 数值形式的类别变量

部分字段虽然在 CSV 中表现为数值，但实际含义是类别。例如：

- `MSSubClass`
- `MoSold`
- `YrSold`

这些字段被转换为字符串，以避免模型误认为类别编号之间存在连续数值距离。

### 4.6.2 有序类别编码

质量类字段具有明确顺序，例如：

```text
Ex > Gd > TA > Fa > Po > None
```

因此，本项目将其映射为数值等级：

```text
Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
```

应用该策略的字段包括 `ExterQual`、`ExterCond`、`BsmtQual`、`BsmtCond`、`HeatingQC`、`KitchenQual`、`FireplaceQu`、`GarageQual`、`GarageCond`、`PoolQC` 等。

此外，`LotShape`、`LandSlope`、`BsmtExposure`、`BsmtFinType1`、`Functional`、`GarageFinish`、`PavedDrive` 等字段也根据其等级含义进行了有序映射。

### 4.6.3 无序类别 one-hot 编码

对没有明确顺序的普通类别变量，使用 one-hot 编码。为了保证训练集和测试集列空间一致，编码前先将训练集和测试集合并，完成缺失值处理、类别转换和编码后，再拆分为训练矩阵和测试矩阵。

## 4.7 偏态数值变量处理

对于右偏较强、取值非负、唯一值数量较多的数值变量，使用 `log1p` 变换。正式建模中共有 27 个数值列执行了偏态修正，例如：

- `LotFrontage`
- `LotArea`
- `MasVnrArea`
- `TotalBsmtSF`
- `GrLivArea`
- `TotalSF`
- `TotalLivingSF`
- `TotalFinishedSF`

该处理降低了极端值对模型训练的影响，使线性模型、核方法和距离敏感模型更加稳定。

引用材料：

- `reports/final_report/tables/skew_transformed_columns.csv`
- `reports/eda/tables/numeric_skewness.csv`

## 4.8 数值缩放

在 one-hot 编码和数值变换完成后，所有数值列使用 `RobustScaler` 进行缩放。与均值方差标准化相比，`RobustScaler` 使用中位数和四分位距，对异常值更稳健。

这一步对 Ridge、ElasticNet、SVR、Kernel Ridge 等模型尤其重要，因为这些模型对特征尺度较敏感。经过缩放后，不同量纲的面积、年份、计数和交互特征可以更稳定地进入同一个模型。

## 4.9 预处理结果

删除 2 个异常训练样本后，正式建模使用的数据规模如下：

| 项目 | 数值 | 说明 |
| --- | ---: | --- |
| 原始训练样本数 | 1460 | 原始训练集 |
| 删除异常点后训练样本数 | 1458 | 正式建模使用 |
| 测试样本数 | 1459 | Kaggle 测试集 |
| 原始特征数 | 80 | 不含 `SalePrice` |
| 高级预处理后基础特征数 | 447 | 不含后期 TE 特征 |
| 缺失指示器特征数 | 34 | `_WasMissing` 特征 |
| 偏态修正数值列数 | 27 | 使用 `log1p` |

引用材料：

- `reports/final_report/tables/preprocessing_feature_summary.csv`

## 4.10 本章小结

本项目的数据预处理不是简单的缺失值填充和编码转换，而是基于 EDA 和字段业务含义进行的系统处理。预处理阶段完成了异常值删除、目标变量变换、语义化缺失值填充、缺失指示器构造、类别变量编码、偏态修正和稳健缩放，为后续特征工程和多模型训练提供了稳定的数据基础。

## 4.11 本章引用材料

- `house_price/data.py`
- `house_price/preprocessing.py`
- `house_price/advanced_preprocessing.py`
- `reports/final_report/tables/preprocessing_feature_summary.csv`
- `reports/final_report/tables/skew_transformed_columns.csv`
