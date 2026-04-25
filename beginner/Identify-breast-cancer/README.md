# 乳腺癌检测项目

## 1. 项目介绍

### 1.1 项目名称

乳腺癌检测（Identify-breast-cancer）

### 1.2 项目目标

本项目旨在使用逻辑回归算法对乳腺癌数据集进行分类，通过建立模型预测肿瘤为良性或恶性，帮助初学者理解机器学习在医疗检测领域的应用。

> ps：这是我的第二个机器学习项目，在实践过程中发现了数据泄露问题并进行了修复，加深了对机器学习流程的理解。

### 1.3 数据集信息

- **数据集名称**：乳腺癌数据集（Breast Cancer Dataset）
- **数据来源**：`sklearn.datasets.load_breast_cancer()`
- **数据规模**：569个样本
- **特征维度**：30个特征
  - 均值特征（10个）：radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
  - 误差特征（10个）：上述特征的误差值
  - 最差特征（10个）：上述特征的最大值
- **类别数量**：2个类别
  - 0：恶性（Malignant）
  - 1：良性（Benign）

### 1.4 运行环境要求

- Python 3.7+
- 依赖包：见 `requirements.txt`

### 1.5 基本使用方法

```bash
# 安装依赖
pip install -r requirements.txt
# 运行模型训练
jupyter lab train_model.ipynb
```

## 2. 项目结构

```bash
Identify-breast-cancer/
├── train_model.ipynb    # 模型训练笔记本
├── requirements.txt     # 项目依赖
└── README.md             # 项目说明文档
```

## 3. 项目收获

### 3.1 了解了机器学习的基本流程

1. 加载数据
2. 数据预处理
3. 分离特征 X 和标签 y
4. 划分训练集、验证集和测试集
5. 训练模型
6. 预测与评估
7. 分析评估指标

### 3.2 为什么选择逻辑回归模型？

- 输出概率值，解释性强，可以知道模型对预测结果的置信度
- 线性分类器，训练速度快，适合初学者理解分类原理
- 在二分类问题中表现稳定，是医疗检测的经典 baseline 模型

### 3.3 数据泄露问题与修复

**问题描述**：

在最初的实现中，StandardScaler 在数据分割之前就对全部数据进行了 fit：

```python
# 错误的做法（数据泄露）
X = data.drop('target', axis=1)
y = data['target']
X = scaler.fit_transform(X)  # 在分割前对全部数据 fit
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4)
```

这会导致验证集和测试集的统计信息（均值、标准差）泄露到训练过程中，使模型评估结果不准确。

**正确做法**：

```python
# 正确的做法（无数据泄露）
X = data.drop('target', axis=1)
y = data['target']
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 只在训练集上 fit
X_cv = scaler.transform(X_cv)             # 用训练集的 scaler transform
X_test = scaler.transform(X_test)          # 用训练集的 scaler transform
```

**经验总结**：
- `fit_transform(X)` = 计算参数 + 转换数据
- `transform(X)` = 只用已计算的参数转换数据
- scaler 必须只在训练集上 fit，验证集和测试集只能 transform

### 3.4 更多评估指标的理解

对于医疗检测场景，仅仅准确率是不够的，需要关注：

- **混淆矩阵**：直观展示分类结果的真假阳性/阴性
- **Precision（精确率）**：预测为恶性的样本中，真正是恶性的比例
- **Recall（召回率）**：所有恶性样本中，被正确识别的比例
- **F1-score**：精确率和召回率的调和平均数

### 3.5 类别分布的重要性

在训练前检查类别分布可以：
- 判断数据集是否平衡
- 选择合适的评估策略
- 避免模型偏向多数类

## 4. 运行结果示例

### 模型评估结果

```
===== 类别分布检查 =====
类别 0 (恶性): 212 样本 (37.3%)
类别 1 (良性): 357 样本 (62.7%)

===== 模型评估结果 =====
训练集准确率: 0.9853
验证集准确率: 0.9912

===== 测试集评估结果 =====
混淆矩阵:
[[43  2]
 [ 1 68]]

分类报告:
              precision    recall  f1-score   support

           0       0.98      0.96      0.97        45
           1       0.97      0.99      0.98        69

    accuracy                           0.97       114
```

**结果解读**：
- 测试集准确率达到 97.37%
- 恶性（0）召回率 96%，表示100个恶性中能识别96个
- 良性（1）召回率 99%，表示100个良性中能识别99个

## 5. TODO

- [ ] 尝试其他分类算法（SVM、决策树、随机森林）
- [ ] 使用交叉验证更稳健地评估模型
- [ ] 添加 ROC 曲线和 AUC 指标
- [ ] 进行超参数调优（正则化参数 C）
- [ ] 部署应用：将模型部署为简单的 Web 应用

## 6. 总结

本项目作为第二个机器学习项目，在第一个项目的基础上加深了对机器学习流程的理解。通过实践，深刻认识到数据泄露问题的严重性，以及数据预处理中"先分割再 fit"的重要性。

在医疗检测场景中，仅仅追求高准确率是不够的，需要综合考虑精确率、召回率等指标，确保模型能够有效地识别恶性肿瘤，避免漏诊。

通过本项目的学习，为后续更复杂的机器学习项目打下了坚实的基础。
