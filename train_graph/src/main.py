import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='evidence', help='which dataset to use')
parser.add_argument('--dim', type=int, default=6, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=10, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=10, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default="plus",
                    help='how to update item at the end of each hop')#:"replace_transform"
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--shows', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')


args = parser.parse_args()
show_loss = False
data_info = load_data(args)
train(args, data_info, show_loss)
