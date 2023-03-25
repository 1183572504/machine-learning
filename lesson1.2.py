from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 导入数据集
iris = load_iris()

# 可视化数据
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
for i in range(iris.data.shape[1]):
    axs[i].scatter(iris.data[iris.target == 0, i], iris.data[iris.target == 0, (i+1) % 4], label=iris.target_names[0], alpha=0.5)
    axs[i].scatter(iris.data[iris.target == 1, i], iris.data[iris.target == 1, (i+1) % 4], label=iris.target_names[1], alpha=0.5)
    axs[i].set_xlabel(iris.feature_names[i])
    axs[i].set_ylabel(iris.feature_names[(i+1) % 4])
    axs[i].legend()
plt.show()

# 定义数据集划分方法
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# 尝试不同的数据集划分比例，并画出性能随比例的变化图
ratios = np.linspace(0.1, 0.9, num=9)
results = []
for ratio in ratios:
    X_train, X_test, y_train, y_test = split_data(iris.data, iris.target, test_size=ratio)
    # 归一化数据
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 建立逻辑回归模型
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append(accuracy)
plt.plot(ratios, results, '-o')
plt.xlabel('Test set ratio')
plt.ylabel('Accuracy')
plt.show()

# 尝试不同的归一化方法，并打印出不同方法的预测结果
scalers = [MinMaxScaler(), StandardScaler()]
for scaler in scalers:
    X_train, X_test, y_train, y_test = split_data(iris.data, iris.target, test_size=0.2)
    # 归一化数据
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 建立逻辑回归模型
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("归一化方法:", scaler.__class__.__name__, "预测准确率:", accuracy)
