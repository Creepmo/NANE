# NANE：Neural-based Attributed Network Embedding

NANE的实现版本

## 使用方法

```python
cd Alg
python main.py -options
```

#### 基本参数选择介绍

- --data，输入数据集名称，以.mat形式存储，位置在./Alg/data下，初始值设为Hamilton，原有数据集选项：[Rochester, Hamilton, facebook]；

- --beta，调整重构非零元素的权重参数，初始值设置为100；

- --alpha，全局损失函数权重，初始值设为500；

- --gamma，属性局部损失函数权重，初始值设为1；

- -zeta，结构局部损失函数权重，初始值设为1；

- --theta，输入属性信息权重，初始值设为1.5；

- --batch_size，每批次训练样本个数，初始值设为100；

- --t，自前馈层输出层维度，初始值设为256；

- --embdim，生成节点隐式向量的维度，初始值设为256；

- --epoch_num，训练周期数，初始值设为200；

- ---learning_rate，学习率，初始值设为0.01。

#### 输入数据格式

以.mat格式存储，包含三个矩阵：Attributes，Label，Network
- Attributes：属性矩阵，每一行表示一个节点的属性，大小为节点个数*属性维度；

- Label：标签矩阵，每一行表示一个节点的标签，大小为节点个数*1；

- Network：结构矩阵，即邻接矩阵，大小为节点个数*节点个数。

#### 输出数据格式

以.mat格式存储，位于./Alg/emb下，以“数据集.emb.mat"命名，包含两个矩阵：embedding，Label

- embedding：节点的隐式表示矩阵，每一行对应一个节点的隐式向量表示，大小为节点个数*表示维度；

- Label：与输入矩阵相同

#### 主要源文件介绍

- main.py：主函数；
- graph.py：图数据预处理模块；
- nane.py：训练函数与评估函数。

#### 评估函数

- node_classify()：节点分类实验；
- node_clustering(): 节点聚类实验。