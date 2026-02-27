import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# 1. 读取数据
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

X_train_raw = train_df.drop(columns=['PRICE']).values
y_train = train_df['PRICE'].values
X_test_raw = test_df.drop(columns=['ID']).values

# 2. 自动化特征工程：二阶交互特征 (让 13 个特征两两相乘，挖掘非线性关系)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = poly.fit_transform(X_train_raw)
X_test = poly.transform(X_test_raw)

print(f"特征数量从 {X_train_raw.shape[1]} 扩充到了 {X_train.shape[1]}")

# 3. 5折交叉验证配置
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(X_test))

# LightGBM 参数
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    # 构建 LightGBM 数据集
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    # 训练并早停 (100轮不降则停)
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_preds = model.predict(X_va, num_iteration=model.best_iteration)
    fold_rmse = np.sqrt(mean_squared_error(y_va, val_preds))
    print(f"Fold {fold + 1} 最佳迭代次数: {model.best_iteration}, 验证集 RMSE: {fold_rmse:.4f}")

    # 累加测试集预测
    test_predictions += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits

# 4. 生成提交文件
sample_df = pd.read_csv('SampleSubmission.csv')
pd.DataFrame({'ID': sample_df['ID'], 'value': test_predictions}).to_csv('submission_lgb_poly.csv', index=False)
print("✅ 生成 LightGBM + 特征交叉的提交文件 submission_lgb_poly.csv")