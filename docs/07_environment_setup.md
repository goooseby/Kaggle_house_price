# 07. Environment Setup

## 当前已有基础环境

当前 baseline 已经可以使用以下依赖运行：

- pandas
- numpy
- scikit-learn

这些足够运行第一版 baseline。

## 下一轮优化建议安装

为了做完整优化回合，建议安装：

```powershell
pip install scipy xgboost lightgbm catboost optuna matplotlib seaborn joblib tqdm
```

或者使用文件安装：

```powershell
pip install -r requirements-optimization.txt
```

如果你使用 Conda，也可以优先装通用包：

```powershell
conda install -c conda-forge scipy matplotlib seaborn tqdm joblib
pip install xgboost lightgbm catboost optuna
```

## 依赖用途

| 依赖 | 用途 |
| --- | --- |
| `scipy` | 偏态变换、权重优化、统计工具 |
| `xgboost` | 强表格模型 |
| `lightgbm` | 强表格模型 |
| `catboost` | 强表格模型，对类别关系友好 |
| `optuna` | 自动调参 |
| `matplotlib` | 可视化 |
| `seaborn` | 可视化 |
| `joblib` | 保存模型和中间结果 |
| `tqdm` | 进度条 |

## 安装后验证

安装完成后，可以运行：

```powershell
python -c "import xgboost, lightgbm, catboost, optuna, scipy; print('optimization env ok')"
```

如果这条命令成功，下一轮完整优化可以直接开始。

## 说明

这道题数据量很小，所以不需要 GPU。

建议优先保证 CPU 版本安装稳定：

- XGBoost CPU 版即可
- LightGBM CPU 版即可
- CatBoost CPU 版即可

如果安装 LightGBM 遇到编译问题，优先用 pip wheel；如果仍然失败，可以先跳过 LightGBM，用 XGBoost + CatBoost + sklearn 模型完成本轮优化。
