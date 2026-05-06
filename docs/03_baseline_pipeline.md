# 03. Baseline Pipeline

## 当前代码结构

项目当前核心代码结构：

```text
main.py
house_price/
  __init__.py
  config.py
  data.py
  eda.py
  preprocessing.py
  modeling.py
outputs/
  eda_report.md
  cv_results.csv
  submission_baseline.csv
```

## main.py 做什么

`main.py` 是总入口，按顺序执行：

1. 读取原始数据
2. 生成 EDA 报告
3. 删除已知异常点
4. 构建训练和测试特征矩阵
5. 训练多个模型并做交叉验证
6. 融合模型预测测试集
7. 输出 Kaggle 提交文件

运行方式：

```powershell
python main.py
```

## 数据读取

由 `house_price/data.py` 负责。

读取文件：

- `train.csv`
- `test.csv`
- `sample_submission.csv`

同时在训练前删除两个经典异常点：

```text
GrLivArea > 4000 且 SalePrice < 300000
```

删除后训练集从 1460 行变为 1458 行。

## 缺失值处理

由 `house_price/preprocessing.py` 负责。

主要策略：

### 1. 设施不存在类字段填 None

例如：

- `PoolQC`
- `MiscFeature`
- `Alley`
- `Fence`
- `FireplaceQu`
- `GarageType`
- `GarageFinish`
- `GarageQual`
- `GarageCond`
- `BsmtQual`
- `BsmtCond`
- `BsmtExposure`
- `BsmtFinType1`
- `BsmtFinType2`

这些字段缺失多半表示“没有这个设施”。

### 2. 面积和数量字段填 0

例如：

- `GarageArea`
- `GarageCars`
- `TotalBsmtSF`
- `BsmtFinSF1`
- `BsmtFinSF2`
- `BsmtUnfSF`
- `BsmtFullBath`
- `BsmtHalfBath`
- `MasVnrArea`

如果没有车库、地下室或贴面，填 0 比填均值更合理。

### 3. `LotFrontage` 按街区中位数填充

临街宽度和街区有关，所以使用：

```text
Neighborhood 分组中位数
```

### 4. 少量类别缺失填众数

例如：

- `MSZoning`
- `Electrical`
- `KitchenQual`
- `Exterior1st`
- `Exterior2nd`
- `SaleType`
- `Utilities`

## 特征工程

当前 baseline 构造了这些派生特征：

| 特征 | 含义 |
| --- | --- |
| `TotalSF` | 地下室 + 一楼 + 二楼总面积 |
| `TotalBathrooms` | 全卫 + 半卫 + 地下室卫浴 |
| `TotalPorchSF` | 各类门廊、露台面积合计 |
| `HouseAge` | 销售年份 - 建造年份 |
| `RemodAge` | 销售年份 - 翻新年份 |
| `GarageAge` | 销售年份 - 车库建造年份 |
| `IsRemodeled` | 是否翻新过 |
| `HasGarage` | 是否有车库 |
| `HasBsmt` | 是否有地下室 |
| `HasFireplace` | 是否有壁炉 |
| `HasPool` | 是否有泳池 |
| `Has2ndFloor` | 是否有二楼 |

这些特征的目的不是炫技，而是把常见业务概念显式告诉模型。

## 类别编码

当前做法：

1. 将训练集和测试集合并预处理
2. 对类别字段做 one-hot encoding
3. 再拆回训练矩阵和测试矩阵

这样可以避免训练集和测试集编码后列不一致。

## 数值缩放

当前使用 `RobustScaler`。

理由：

- 房价数据有异常值和长尾特征
- `RobustScaler` 使用中位数和四分位距
- 相比普通标准化，对异常值更稳

## 模型

当前 baseline 使用：

- Ridge
- Lasso
- ElasticNet
- GradientBoostingRegressor
- RandomForestRegressor

最终提交使用四个表现较好的模型做简单平均：

- Ridge
- Lasso
- ElasticNet
- GradientBoostingRegressor

RandomForest 当前表现较弱，没有进入最终 blend。
