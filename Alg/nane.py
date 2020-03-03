#coding:utf-8

import tensorflow as tf
from tfrbm import BBRBM, GBRBM
import numpy as np
import random
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.cluster import KMeans

class NANE(object):
    def __init__(self,graph,args):
        self.graph = graph
        self.RBM_init = False # 是否使用RBM预训练W、b等参数
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.embdim = args.embdim
        self.embpath = 'emb/'+args.data+'.emb'
        self.theta = args.theta

        #定义autoencoder结构
        self.attention_x = args.t
        self.attention_a = args.t
        self.attr_layers = [self.attention_x+self.attention_a,args.embdim,graph.adjmat.shape[1]+graph.attrmat.shape[1]]
        attr_layer_length = len(self.attr_layers)
        self.attr_size = graph.attrmat.shape[1]
        self.struc_size = graph.adjmat.shape[1]
        
        self.attr_mid_layer = (attr_layer_length - 1)/2

        self.X = tf.placeholder(tf.float32,[None,self.struc_size])
        self.A = tf.placeholder(tf.float32,[None,self.attr_size])
        self.adjmat = tf.placeholder(tf.float32,[None,None])
        self.attrmat = tf.placeholder(tf.float32,[None,None])

        with tf.name_scope("attr_emb"):
            self.attr_W = []
            self.attr_b = []
            self.attr_hidden = []
            attr_reg_loss = 0.0
            cur_seed = random.getrandbits(32)
            self.W_attx = tf.get_variable(name = "W_attx",shape=[self.struc_size,self.attention_x],initializer = tf.contrib.layers.xavier_initializer(uniform=False,seed= cur_seed))
            self.b_attx = tf.Variable(name="b_attx",initial_value=tf.zeros([self.attention_x]))
            self.W_atta = tf.get_variable(name = "W_atta",shape=[self.attr_size,self.attention_a],initializer = tf.contrib.layers.xavier_initializer(uniform=False,seed= cur_seed))
            self.b_atta = tf.Variable(name="b_attx",initial_value=tf.zeros([self.attention_a]))
            self.X_attention = tf.nn.tanh(tf.nn.dropout(tf.matmul(self.X,self.W_attx)+self.b_attx,keep_prob=0.7))
            self.A_attention = tf.nn.tanh(tf.nn.dropout(tf.matmul(self.A,self.W_atta)+self.b_atta,keep_prob=0.7))
            attr_reg_loss += tf.nn.l2_loss(self.W_attx)+tf.nn.l2_loss(self.b_attx)
            attr_reg_loss += tf.nn.l2_loss(self.W_atta)+tf.nn.l2_loss(self.b_atta)
            self.S = tf.concat([self.X,self.theta*self.A],axis=1)
            self.S_attention = tf.concat([self.X_attention,self.theta*self.A_attention],axis=1)
            for i in range(attr_layer_length-1):
                cur_seed = random.getrandbits(32)
                self.attr_W.append(tf.get_variable(name = "attr_W"+str(i),shape=[self.attr_layers[i],self.attr_layers[i+1]],initializer = tf.contrib.layers.xavier_initializer(uniform=False,seed= cur_seed)))
                self.attr_b.append(tf.Variable(name="attr_b"+str(i),initial_value=tf.zeros([self.attr_layers[i+1]])))
                attr_reg_loss += tf.nn.l2_loss(self.attr_W[i])+tf.nn.l2_loss(self.attr_b[i])
                #第一层与第二层之间的连接
                if(i==0):
                    attr_layer = tf.nn.tanh(tf.matmul(self.S_attention,self.attr_W[i])+self.attr_b[i])
                #倒数第二层和最后一层之间的连接用sigmoid激活
                elif(i==attr_layer_length-2):
                    attr_layer = tf.nn.sigmoid(tf.matmul(self.attr_hidden[i-1],self.attr_W[i])+self.attr_b[i])
                else:
                    attr_layer = tf.nn.tanh(tf.matmul(self.attr_hidden[i-1],self.attr_W[i])+self.attr_b[i])

                if(i==(attr_layer_length-3)/2):
                    self.attr_node_emb = attr_layer
                    self.attr_hidden.append(self.attr_node_emb)
                else:
                    self.attr_hidden.append(attr_layer)

        
        attr_B = self.S * (args.beta -1) + 1
        attr_2nd_loss = tf.reduce_sum(tf.pow((self.attr_hidden[-1]-self.S)*attr_B,2))
        struc_D = tf.diag(tf.reduce_sum(self.adjmat,1))
        struc_L = struc_D - self.adjmat
        struc_1st_loss = 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(self.attr_node_emb),struc_L),self.attr_node_emb))
        attr_D = tf.diag(tf.reduce_sum(self.attrmat,1))
        attr_L = attr_D - self.attrmat
        attr_1st_loss = 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(self.attr_node_emb),attr_L),self.attr_node_emb))
        self.loss = args.gamma*attr_1st_loss + args.alpha*attr_2nd_loss + args.zeta*struc_1st_loss + args.reg*attr_reg_loss
        self.optimizer = tf.train.RMSPropOptimizer(args.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        
    # 进行训练的函数
    def train(self):

        if self.RBM_init:
            attr_bbrbm = BBRBM(n_visible=self.attr_size, n_hidden=self.embdim, learning_rate=0.01, momentum=0.95, use_tqdm=True)
            attr_bbrbm.fit(self.graph.attrmat, n_epoches=100, batch_size=500, shuffle=True, verbose=True)
            attr_W, attr_Bv, attr_Bh = attr_bbrbm.get_weights()

        init = tf.global_variables_initializer()
        nodes_emb = np.zeros([self.struc_size,self.embdim])
        min_loss = np.inf
        with tf.Session() as sess:
            sess.run(init)
            if self.RBM_init:
                sess.run(self.attr_W[0].assign(attr_W))
                sess.run(self.attr_b[0].assign(attr_Bh))
                sess.run(self.attr_W[1].assign(tf.transpose(attr_W)))
                sess.run(self.attr_b[1].assign(attr_Bv))
            for epoch in range(self.epoch_num):
                nodes_emb_tmp = None
                loss = 0.0
                batches = self.graph.mini_batch()
                for batch_id,batch in enumerate(batches):
                    X_batch,adjmat_batch,attrX_batch,attrmat_batch = batch
                    feed_dict = {self.X:X_batch,self.A:attrX_batch,self.adjmat:adjmat_batch,self.attrmat:attrmat_batch}
                    _,batch_emb,batch_loss = sess.run([self.optimizer,self.attr_node_emb,self.loss],feed_dict=feed_dict)
                    loss += batch_loss
                    if nodes_emb_tmp is None:
                        nodes_emb_tmp = batch_emb
                    else:
                        nodes_emb_tmp = np.vstack((nodes_emb_tmp,batch_emb))
                if(loss < min_loss):
                    print("epoch:%3d\tloss:%.2f\tsave the best result."%(epoch,loss))
                    for i,node_emb in enumerate(nodes_emb_tmp):
                        sample_node = self.graph.indices[i]
                        nodes_emb[sample_node] = node_emb
                    sio.savemat(self.embpath,{'embedding':nodes_emb,'label':self.graph.label})
                    min_loss = loss
                else:
                    print("epoch:%3d\tloss:%.2f\t"%(epoch,loss))
# 测试函数类
class EVAL(object):

    def __init__(self,args):
    	self.embpath = 'emb/'+args.data+'.emb'

    # 节点分类函数 输出F1-score(micro)
    def node_classify(self):
        data = sio.loadmat(self.embpath)
        label = data['label'].ravel()
        emb = data['embedding']
        micro_list = []
        # macro_list = []
        for test_size in [0.85,0.75,0.65,0.55,0.45,0.35,0.25]:
            x_train, x_test, y_train, y_test = train_test_split(emb, label, random_state=1, test_size=test_size)
            clf = svm.SVC(C=100,kernel='rbf')
            clf.fit(x_train,y_train)
            y_train_hat = clf.predict(x_train)
            y_test_hat = clf.predict(x_test)
            micro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='micro')*10000)/100))
            # macro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='macro')*10000)/100))
        print("node classfication...")
        print("F1-score(micro): %s"%(" ".join(micro_list)))
        # print("F1-score(macro): %s"%(" ".join(macro_list)))

	# 节点聚类函数 输出adjusted_rand_score和normalized_mutual_info_score
    def node_clustering(self):
        data = sio.loadmat(self.embpath)
        label = data['label'].ravel()
        emb = data['embedding']
        clf = KMeans(n_clusters=4,init="k-means++")
        kmeans = clf.fit(emb)
        cluster_groups = kmeans.labels_
        acc = metrics.adjusted_rand_score(label,cluster_groups)
        nmi = metrics.normalized_mutual_info_score(label,cluster_groups)
        
        print("node clustering...")
        print("acc: %.4f, nmi: %.4f"%(acc,nmi))