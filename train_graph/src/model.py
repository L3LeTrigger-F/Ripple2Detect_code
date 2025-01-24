import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()
# 使用方法

class RippleNet(object):
    def __init__(self, args, n_entity, n_relation):
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):#每一个hop建立？？
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))
    def laplace_initializer(self, loc, scale):
        def _initializer(shape, dtype=None, partition_info=None):
            # 生成拉普拉斯分布的随机数
            u = tf.random.uniform(shape, minval=-0.5, maxval=0.5, dtype=dtype or tf.float64)
            return loc - scale * tf.sign(u) * tf.math.log(1.0 - 2.0 * tf.abs(u))
        return _initializer
    
    def t_distribution_initializer(self, loc, scale, df):
        def _initializer(shape, dtype=None, partition_info=None):
            # 生成标准正态分布的随机数 Z
            z = tf.random.normal(shape, dtype=dtype or tf.float64)
            # 生成卡方分布的随机数 V (自由度 df)
            v = tf.random.gamma(shape, alpha=[df / 2], dtype=dtype or tf.float64) * 2
            # 生成 t 分布的随机数 t = Z / sqrt(V / df)
            v_squeezed = tf.squeeze(v)  # 去掉多余的维度
            t_dist = z / tf.sqrt(v_squeezed / df)
            # 使用给定的 loc 和 scale 调整 t 分布
            return loc + scale * t_dist
    
        return _initializer
    def _build_embeddings(self):
        #laplace_initializer_entity = self.laplace_initializer(loc=0.0, scale=0.1)
        #laplace_initializer_relation = self.laplace_initializer(loc=0.0, scale=0.1)
        t_initializer_entity=self.t_distribution_initializer(loc=0.0, scale=0.1,df=5.0)
        t_initializer_relation = self.t_distribution_initializer(loc=0.0, scale=0.1,df=5.0)
        self.entity_emb_matrix = tf.get_variable(name="entity_emb_matrix", dtype=tf.float64,
                                                 shape=[self.n_entity, self.dim],
                                                 #initializer=t_initializer_entity)
                                                 #initializer=t_distribution_initializer)
                                                # initializer=laplace_initializer_entity)
                                                  initializer=tf.keras.initializers.glorot_normal())
        self.relation_emb_matrix = tf.get_variable(name="relation_emb_matrix", dtype=tf.float64,
                                                   shape=[self.n_relation, self.dim, self.dim],
                                                  # initializer=t_initializer_relation)
                                                   #initializer=laplace_initializer_relation)
                                                   initializer=tf.keras.initializers.glorot_normal())

    def _build_model(self):
        #initializer_laplace = self.laplace_initializer(loc=0.0, scale=0.1)
        # transformation matrix for updating item embeddings at the end of each hop #(16,16)
        t_initializer_entity=self.t_distribution_initializer(loc=0.0, scale=0.1,df=5.0)
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.keras.initializers.glorot_normal())
                                               #initializer=t_initializer_entity)
                                               #initializer=initializer_laplace)

        # [batch size, dim] # (?,16)
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):#2
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        ori_time=time.time()
        pre_time=ori_time
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
                ss_time=time.time()
                print(ss_time-pre_time)
                pre_time=ss_time
        print(time.time()-ori_time)
        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, clip_norm=5)
                     for gradient in gradients]
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables))
        '''
    def cal_attack(self,predictions,labels,items,val):
        tp=sum(1 for p, l,x in zip(predictions, labels,items) if p == 1 and l == 1 and x == val)
        tn=sum(1 for p, l,x in zip(predictions, labels,items) if p == 0 and l == 0  and x == val)
        fp=sum(1 for p, l,x in zip(predictions, labels,items) if p == 1 and l == 0 and x == val)
        fn=sum(1 for p, l,x in zip(predictions, labels,items) if p == 0 and l == 0  and x == val)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        acc=np.mean(np.equal(predictions, labels))
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        y_truth=[]
        y_pred=[]
        for p,l,x in zip(predictions, labels,items):
            if x==val:
                y_truth.append(l)
                y_pred.append(p)
        if len(y_truth)!=0:
            # auc_score = roc_auc_score(y_truth,y_pred)
            # print("f1_score:",f1_score,"auc:",auc_score,"x:",val)
            print("f1_score:",f1_score,"acc:",acc,"x:",val)
    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)
    def eval(self, sess,model, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except:
            auc=1
        items=feed_dict[model.items]
        # user=
        sort_dict={}
        for i,sco in zip(items,scores):
            sort_dict[i]=sco
        sort_dict=sorted(sort_dict.items(), key=lambda d: d[1], reverse=True)
        # pre_item=sort_dict[0]
        predictions = [1 if i >= 0.1 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        total = len(labels)
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        if tn+fp==0:
            FPR=0
        else:
            FPR=fp/(tn+fp)
        user_tp=sum(1 for p, l,x in zip(predictions, labels,items) if p == 1 and l == 1 and x>=8 and x<=11)
        user_tn = sum(1 for p, l ,x in zip(predictions, labels,items) if p == 0 and l == 0 and x>=8 and x<=11)
        user_fp = sum(1 for p, l ,x in zip(predictions, labels,items) if p == 1 and l == 0 and x>=8 and x<=11)
        user_fn = sum(1 for p, l ,x in zip(predictions, labels,items) if p == 0 and l == 1 and x>=8 and x<=11)
        # predict(self.)
        # user_precision = user_tp / (user_tp + user_fp) if user_tp + user_fp > 0 else 0.0
        # user_recall = user_tp / (user_tp + user_fn) if user_tp + user_fn > 0 else 0.0
        # user_f1_score = 2 * (user_precision * user_recall) / (user_precision + user_recall) if user_precision + user_recall > 0 else 0.0
        # if user_tn+user_fp==0:
            # user_FPR=0
        # else:
            # user_FPR=user_fp/(user_tn+user_fp)
        # for p,l,x in zip(predictions, labels,items):
            # if p == 1 and l == 0:
                # print(x)
        # self.cal_attack(predictions,labels,items,6)
        # self.cal_attack(predictions,labels,items,7)
        # self.cal_attack(predictions,labels,items,8)
        return auc, acc,precision,recall,f1_score,user_tp,user_tn,user_fp,user_fn
