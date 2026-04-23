import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1.加载数据
data_path = 'data/iris.data'
column_names = ['sepal length (cm)', 
                'sepal width (cm)', 
                'petal length (cm)', 
                'petal width (cm)', 
                'species']

df = pd.read_csv(data_path, header=None, names=column_names)

# 2.数据预处理 

# 3. 分离特征 X 和标签 y

X = df.drop('species', axis=1)   # 去掉标签列作为特征
y = df['species']      

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

# 5. 训练模型（KNN）
model = KNeighborsClassifier(n_neighbors=11)
model.fit(X_train , y_train)

# 6. 预测与评估
y_pred = model.predict(X_test)
print("\n准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))

# 7. 预测新样本
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # 按特征顺序填写数值
predicted = model.predict(new_sample)
print(f"\n新样本预测类别: {predicted[0]}")