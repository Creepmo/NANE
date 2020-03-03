#coding:utf-8

import tensorflow as tf
import numpy as np

from graph import Graph
from nane import NANE,EVAL
import argparse
import json

def parse_args(): 
    parser = argparse.ArgumentParser(description="Run NANE.")
    parser.add_argument('--data', nargs='?',default="Hamilton",help="Rochester,Hamilton,facebook")
    parser.add_argument('--beta', type=float ,default=100,\
        help="the weight balanced value to reconstruct non-zero element more.")
    parser.add_argument('--alpha', type=float ,default=500, help="the weight of global reconstruction loss")
    parser.add_argument('--gamma', type=float ,default=1, help="the weight of local attribute pairwise loss")
    parser.add_argument('--zeta', type=float ,default=1, help="the weight of local structure pairwise loss")
    parser.add_argument('--reg', type=float ,default=1, help="the weight of regularization loss")
    parser.add_argument('--theta', type=float ,default=1.5, help="the weight of attribute information")
    parser.add_argument('--batch_size', type=int ,default=100,help="batch size")
    parser.add_argument('--t', type=int ,default=256,help="the unit number of feedforward hidden layer")
    parser.add_argument('--embdim', type=int ,default=256,help="the dimension of embeddings")
    parser.add_argument('--epoch_num', type=int ,default=200,help="the number of epoches")
    parser.add_argument('--learning_rate', type=float ,default=0.01,help="learning rate")

    return parser.parse_args()

def main(args):
    graph = Graph(args)
    nane = NANE(graph,args)
    nane.train()
    eval = EVAL(args)
    eval.node_clustering()
    eval.node_classify()

if __name__ == '__main__':
    args = parse_args()
    main(args)