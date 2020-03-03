#coding:utf-8
import numpy as np
import scipy.io as sio
from scipy.sparse import csr_matrix 
from sklearn import metrics

# graph类：读入数据并产生batch
# 输入数据：mat文件，包含：data['Network']以csr_matrix存储的邻接矩阵，data['Label']每个节点的社区分类
class Graph(object):
    def __init__(self,args):
        data = sio.loadmat('./data/'+args.data+'.mat')
        self.batch_size = args.batch_size
        self.batch_shuffle = True
        self.adjmat = csr_matrix(data['Network']).toarray()
        attr = csr_matrix(data['Attributes']).toarray()
        self.label = data['Label']
        print("calculate cosine similarity..")
        self.attrmat = metrics.pairwise.cosine_similarity(attr,attr)

    # 产生batch便于训练。输出包含邻接矩阵的batch（adjmat_batch），属性相似度矩阵的batch（attrmat_batch）
    # 以及节点结构的global information（X_batch）、属性的global information（attrX_batch）
    def mini_batch(self):
        data_size = len(self.label)
        if(self.batch_shuffle):
            self.indices = np.random.permutation(np.arange(data_size))
        else:
            self.indices = np.arange(data_size)
        start_index = 0
        end_index = min(start_index+self.batch_size,data_size)
        while(start_index < data_size):
            index = self.indices[start_index:end_index]
            adjmat_batch = self.adjmat[index][:,index]
            attrmat_batch = self.attrmat[index][:,index]
            attrX_batch = self.attrmat[index]
            X_batch = self.adjmat[index]
            yield X_batch,adjmat_batch,attrX_batch,attrmat_batch
            start_index = end_index
            end_index = min(start_index+self.batch_size,data_size)