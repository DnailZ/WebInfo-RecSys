
# 文档搜索引擎实践

<center><h5>PB18000028 邱子悦</h5></center>

<center><h5>PB18000058 丁垣天</h5></center>

[TOC]

## 1 实验目的

通过推荐系统实践，学习信息挖掘技术。

## 2 实验简介

数据来源为豆瓣电影的评分记录。 根据训练数据中的用户评分信息，判断用户偏好，并为测试数据中 user-item 对进行评分。

必备考核内容:个性化推荐技术(方法自选)。

可选考核内容:标签处理(NLP 技术)、社交推荐(社交网络分析技术)。

基于评分预测的 RMSE 指标进行评估。

## 3 实验说明



## 4 实验过程

### 4.1 Bias-SVD

推荐系统中最为主流与经典的技术之一是协同过滤技术，可根据是否采用了机器学习思想建模的不同划分为基于内存的协同过滤（Memory-based CF）与基于模型的协同过滤技术（Model-based CF）。基于模型的协同过滤技术中尤为矩阵分解（Matrix Factorization）技术最为普遍和流行，因为它的**可扩展性极好并且易于实现**。

#### 4.1.1 矩阵分解

评分预测的思路是将已有的评分矩阵 (items - users) 拆成 (items - factors) * (factors - users)，揭示潜在因子。当用户与物品的潜在因子已知，则任何缺失的评分，均可以通过分解出的两个矩阵行列运算得到。这利用的便是矩阵分解技术，其中使用广泛的是奇异值分解 (SVD)。

但是，SVD 要求矩阵是稠密的。直接用传统 SVD 算法并不是一个好选择。

#### 4.1.2 Funk-SVD

Funk-SVD 借鉴线性回归的思想，通过最小化观察数据的平方来寻求最优的用户和项目的隐含向量表示。这种方法被称之为隐语义模型 (Latent factor model, LFM)，其算法意义层面的解释为通过隐含特征 (latent factor) 将 user 兴趣与 item 特征联系起来。

#### 4.1.3 Bias-SVD

在 Funk-SVD 提出来之后，陆续又提出了许多变形版本，其中相对流行的方法是 Bias-SVD。

对于一个评分系统有些固有属性和用户物品无关，而用户也有些属性和物品无关，物品也有些属性与用户无关，具体的预测公式如下：

![biasSVD](pics/biasSVD.png)

其中 $\mu$ 为整个网站的平均评分； $b_u$ 为用户的评分偏置，$b_i$ 为项目的被评分偏置。

#### 4.1.4 用 embedding 的方法实现矩阵分解

将用户和电影通过 embedding 层压缩到 k 维度向量，然后进行向量点乘，得到用户对电影的预测评分。

参考架构图如下：

![embedding](pics/embedding.jpg)

我们使用 pytorch 来实现：

```python
class DualEmbedding(nn.Module):
    def __init__(self, user_n, movie_n, k):
        super(DualEmbedding, self).__init__()
        self.user_embed = nn.Embedding(user_n, k)
        self.user_bias = nn.Embedding(user_n, 1)
        self.movie_embed = nn.Embedding(movie_n, k)
        self.movie_bias = nn.Embedding(movie_n, 1)
    
    def forward(self, user, movie):
        user_feat = self.user_embed(user)
        movie_feat = self.movie_embed(movie)
        dot_product = torch.sum(user_feat * movie_feat, dim=-1)
        result = dot_product + user_bias + movie_bias
        return (torch.sigmoid(dot_product), self.l1_loss())
```

### 4.2 算法实现

#### 4.2.1 自动微分梯度下降

#### 4.2.2 正则化

### 4.3 利用社交网络和标签信息

#### 4.3.1 社交约束

[SoRec](https://www.researchgate.net/publication/221615498_SoRec_Social_recommendation_using_probabilistic_matrix_factorization)

#### 4.3.2 利用标签信息


## 5 实验结果

### 5.1 梯度下降效果

### 5.2 RMSE

### 5.3 部分结果分析

