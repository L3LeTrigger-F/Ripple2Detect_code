import tensorflow as tf
import numpy as np
from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]+1
    n_relation = data_info[4]
    ripple_set = data_info[5]
    # show_indices=data_info[6]

    model = RippleNet(args, n_entity, n_relation)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:#(452,3)
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc,train_pre,train_rec,train_f1= evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc,eval_pre,eval_rec,eval_f1 = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
           # test_auc, test_acc,test_pre,test_rec,test_f1 = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            # show_data=np.array([[2002,0,1],[2002,1,1],[2002,2,1],[2002,3,1],[2002,4,1],[2002,5,1],[2002,6,1],[2002,7,1],[2002,8,1],
                                # [2002,9,1],[2002,10,1],[2002,11,1],[2002,12,1],[2002,13,1],[2002,14,1],[2002,15,1],[2002,16,1]
                                # ])
            # show_data=np.array([[2002,3,1],[2002,4,1],[2002,5,1],[2002,9,1],[2002,0,0],[2002,1,0],[2002,2,0],[2002,6,0],[2002,7,0],[2002,8,0],[2002,10,0]])
            # show_data=np.array([[2004,3,1],[2004,4,1],[2004,5,1],[2004,9,1],[2004,0,0],[2004,1,0],[2004,2,0],[2004,6,0],[2004,7,0],[2004,8,0],[2004,10,0]])
            '''
            show_data=np.array([[2034,3,1],[2034,4,1],[2034,5,1],[2034,9,1],[2034,0,0],[2034,1,0],[2034,2,0],[2034,6,0],[2034,7,0],[2034,8,0],[2034,10,0],[2034,11,0]])
            xx= evaluation(sess, args, model,show_data, ripple_set, args.batch_size)
            '''
            # model.predict()
            print('epoch %d    train auc: %.4f  train_acc: %.4f train_pre: %.4f train_rec: %.4f train_f1: %.4f'  % (step, train_auc, train_acc,train_pre,train_rec,train_f1))
            print('eval auc: %.4f  eval_acc: %.4f eval_pre: %.4f eval_rec: %.4f eval_f1: %.4f'  % (eval_auc, eval_acc,eval_pre,eval_rec,eval_f1))


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list=[]
    rec_list=[]
    pre_list=[]
    user_tp_all=0
    user_fp_all=0
    user_fn_all=0
    user_tn_all=0
    while start < data.shape[0]:
        auc, acc,pre,rec,f1,user_tp,user_tn,user_fp,user_fn = model.eval(sess,model, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        start += batch_size
        user_tp_all+=user_tp
        user_fp_all+=user_fp
        user_fn_all+=user_fn
        user_tn_all+=user_tn
    user_acc=(user_tp_all+user_tn_all)/(user_tp_all+user_tn_all+user_fp_all+user_fn_all) if(user_tp_all+user_tn_all+user_fp_all+user_fn_all)>0 else 0.0
    user_precision = user_tp_all / (user_tp_all + user_fp_all) if user_tp_all + user_fp_all > 0 else 0.0
    user_recall = user_tp_all / (user_tp_all + user_fn_all) if user_tp_all + user_fn_all > 0 else 0.0
    user_f1_score = 2 * (user_precision * user_recall) / (user_precision + user_recall) if user_precision + user_recall > 0 else 0.0
    if user_tn_all+user_fp_all==0:
        user_FPR=0
    else:
        user_FPR=user_fp_all/(user_tn_all+user_fp_all)
    
    return float(np.mean(auc_list)), float(np.mean(acc_list)),float(np.mean(pre_list)), float(np.mean(rec_list)),float(np.mean(f1_list))
