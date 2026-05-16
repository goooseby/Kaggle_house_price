# Kaggle House Prices EDA Report

## Dataset Shape

- train: 1460 rows x 81 columns
- test: 1459 rows x 80 columns
- train-only columns: ['SalePrice']
- test-only columns: []

## Target Summary

| index | SalePrice |
| --- | --- |
| count | 1460.0 |
| mean | 180921.19589041095 |
| std | 79442.50288288663 |
| min | 34900.0 |
| 25% | 129975.0 |
| 50% | 163000.0 |
| 75% | 214000.0 |
| max | 755000.0 |

- SalePrice skew: 1.8829
- log1p(SalePrice) skew: 0.1213

## Top Missing Values - Train

| index | missing | pct |
| --- | --- | --- |
| PoolQC | 1453 | 99.52 |
| MiscFeature | 1406 | 96.3 |
| Alley | 1369 | 93.77 |
| Fence | 1179 | 80.75 |
| MasVnrType | 872 | 59.73 |
| FireplaceQu | 690 | 47.26 |
| LotFrontage | 259 | 17.74 |
| GarageType | 81 | 5.55 |
| GarageYrBlt | 81 | 5.55 |
| GarageFinish | 81 | 5.55 |
| GarageQual | 81 | 5.55 |
| GarageCond | 81 | 5.55 |
| BsmtFinType2 | 38 | 2.6 |
| BsmtExposure | 38 | 2.6 |
| BsmtFinType1 | 37 | 2.53 |
| BsmtCond | 37 | 2.53 |
| BsmtQual | 37 | 2.53 |
| MasVnrArea | 8 | 0.55 |
| Electrical | 1 | 0.07 |

## Top Missing Values - Test

| index | missing | pct |
| --- | --- | --- |
| PoolQC | 1456 | 99.79 |
| MiscFeature | 1408 | 96.5 |
| Alley | 1352 | 92.67 |
| Fence | 1169 | 80.12 |
| MasVnrType | 894 | 61.27 |
| FireplaceQu | 730 | 50.03 |
| LotFrontage | 227 | 15.56 |
| GarageCond | 78 | 5.35 |
| GarageYrBlt | 78 | 5.35 |
| GarageQual | 78 | 5.35 |
| GarageFinish | 78 | 5.35 |
| GarageType | 76 | 5.21 |
| BsmtCond | 45 | 3.08 |
| BsmtExposure | 44 | 3.02 |
| BsmtQual | 44 | 3.02 |
| BsmtFinType1 | 42 | 2.88 |
| BsmtFinType2 | 42 | 2.88 |
| MasVnrArea | 15 | 1.03 |
| MSZoning | 4 | 0.27 |
| BsmtFullBath | 2 | 0.14 |
| BsmtHalfBath | 2 | 0.14 |
| Functional | 2 | 0.14 |
| Utilities | 2 | 0.14 |
| GarageCars | 1 | 0.07 |
| GarageArea | 1 | 0.07 |
| TotalBsmtSF | 1 | 0.07 |
| KitchenQual | 1 | 0.07 |
| BsmtUnfSF | 1 | 0.07 |
| BsmtFinSF2 | 1 | 0.07 |
| BsmtFinSF1 | 1 | 0.07 |

## Numeric Correlation With SalePrice

| index | corr |
| --- | --- |
| OverallQual | 0.7909816005838053 |
| GrLivArea | 0.7086244776126515 |
| GarageCars | 0.6404091972583519 |
| GarageArea | 0.6234314389183622 |
| TotalBsmtSF | 0.6135805515591943 |
| 1stFlrSF | 0.6058521846919153 |
| FullBath | 0.5606637627484453 |
| TotRmsAbvGrd | 0.5337231555820284 |
| YearBuilt | 0.5228973328794967 |
| YearRemodAdd | 0.5071009671113866 |
| GarageYrBlt | 0.4863616774878596 |
| MasVnrArea | 0.47749304709571444 |
| Fireplaces | 0.46692883675152763 |
| BsmtFinSF1 | 0.3864198062421535 |
| LotFrontage | 0.35179909657067737 |
| WoodDeckSF | 0.32441344456812926 |
| 2ndFlrSF | 0.31933380283206736 |
| OpenPorchSF | 0.31585622711605504 |
| HalfBath | 0.28410767559478256 |
| LotArea | 0.2638433538714051 |

## Categorical Cardinality

| index | nunique |
| --- | --- |
| Neighborhood | 25 |
| Exterior2nd | 16 |
| Exterior1st | 15 |
| SaleType | 9 |
| Condition1 | 9 |
| Condition2 | 8 |
| HouseStyle | 8 |
| RoofMatl | 8 |
| Functional | 7 |
| BsmtFinType2 | 6 |
| Heating | 6 |
| RoofStyle | 6 |
| SaleCondition | 6 |
| BsmtFinType1 | 6 |
| GarageType | 6 |
| Foundation | 6 |
| Electrical | 5 |
| FireplaceQu | 5 |
| HeatingQC | 5 |
| GarageQual | 5 |

## High Median Price Neighborhoods

| Neighborhood | count | median | mean |
| --- | --- | --- | --- |
| NridgHt | 77 | 315000.0 | 316270.6 |
| NoRidge | 41 | 301500.0 | 335295.3 |
| StoneBr | 25 | 278000.0 | 310499.0 |
| Timber | 38 | 228475.0 | 242247.4 |
| Somerst | 86 | 225500.0 | 225379.8 |
| Veenker | 11 | 218000.0 | 238772.7 |
| Crawfor | 51 | 200624.0 | 210624.7 |
| ClearCr | 28 | 200250.0 | 212565.4 |
| CollgCr | 150 | 197200.0 | 197965.8 |
| Blmngtn | 17 | 191000.0 | 194870.9 |
| NWAmes | 73 | 182900.0 | 189050.1 |
| Gilbert | 79 | 181000.0 | 192854.5 |

## Known Outlier Candidates

| index | Id | GrLivArea | SalePrice | OverallQual | Neighborhood |
| --- | --- | --- | --- | --- | --- |
| 523 | 524 | 4676 | 184750 | 10 | Edwards |
| 1298 | 1299 | 5642 | 160000 | 10 | Edwards |

## Notes

- The target is strongly right-skewed, so the modeling pipeline uses log1p(SalePrice).
- Many missing categorical values mean the house lacks that feature, not bad data.
- The two GrLivArea outliers are removed before training.