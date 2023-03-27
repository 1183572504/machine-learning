from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

boston = load_boston()
X = boston.data
y = boston.target


fig, axs = plt.subplots(3, 5, figsize=(20, 15))
axs = axs.ravel()

for i in range(X.shape[1]):
    axs[i].scatter(X[:, i], y, s=5)
    axs[i].set_xlabel(boston.feature_names[i])
    axs[i].set_ylabel('Price')

plt.show()

kf_5 = KFold(n_splits=5, shuffle=True, random_state=42)
kf_10 = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf_5.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



scaler_1 = StandardScaler()
X_scaled_1 = scaler_1.fit_transform(X)

scaler_2 = MinMaxScaler()
X_scaled_2 = scaler_2.fit_transform(X)





lr = LinearRegression()
lr.fit(X_train, y_train)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train)


# 交叉验证评估模型
scores_lr = cross_val_score(lr, X_train, y_train, cv=kf_5, scoring='neg_mean_squared_error')
print('MSE (Linear Regression, 5-fold):', -scores_lr.mean())
scores_sgd = cross_val_score(sgd, X_train, y_train, cv=kf_5, scoring='neg_mean_squared_error')
print('MSE (SGD Regression, 5-fold):', -scores_sgd.mean())

# 预测测试集并评估模型
y_pred_lr = lr.predict(X_test)
print('MSE (Linear Regression, test set):', mean_squared_error(y_test, y_pred_lr))
print('MAE (Linear Regression, test set):', mean_absolute_error(y_test, y_pred_lr))
y_pred_sgd = sgd.predict(X_test)
print('MSE (SGD Regression, test set):', mean_squared_error(y_test, y_pred_sgd))
print('MAE (SGD Regression, test set):', mean_absolute_error(y_test, y_pred_sgd))
