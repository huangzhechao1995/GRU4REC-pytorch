
import numpy as np
import pandas as pd
import argparse
import torch
import lib
import numpy as np
import os
import datetime

from baselines import RandomPred, Pop,  SessionPop, ItemKNN, BPR


import torch


def evaluate_sessions(pr, test_data, train_data, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):    
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.
    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    
    '''
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items): items_to_predict = items
                else: items_to_predict = np.hstack(([iid], items))      
            preds = pr.predict_next(sid, prev_iid, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            rank = (preds > preds[iid]).sum()+1
            assert rank > 0
            if rank < cut_off:
                recall += 1
                mrr += 1.0/rank
            evalutation_point_count += 1
        prev_iid = iid
    return recall/evalutation_point_count, mrr/evalutation_point_count


#-------Data-------
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', default='data/preprocessed_data', type=str)
parser.add_argument('--train_data', default='rsc15_train_full.txt', type=str)
parser.add_argument('--valid_data', default='rsc15_test.txt', type=str)
parser.add_argument('--test_data', default='rsc15_test.txt', type=str)

# Get the arguments
args = parser.parse_args()
print("Loading train data from {}".format(os.path.join(args.data_folder, args.train_data)))
print("Loading valid data from {}".format(os.path.join(args.data_folder, args.valid_data)))
print("Loading test data from {}\n".format(os.path.join(args.data_folder, args.test_data)))

train_data = pd.read_csv(os.path.join(args.data_folder, args.train_data), sep='\t', dtype={'ItemId':np.int64})
test_data = pd.read_csv(os.path.join(args.data_folder, args.test_data), sep='\t', dtype={'ItemId':np.int64})

"""train_data = lib.Dataset(os.path.join(args.data_folder, args.train_data))
valid_data = lib.Dataset(os.path.join(args.data_folder, args.valid_data), itemmap=train_data.itemmap)
test_data = lib.Dataset(os.path.join(args.data_folder, args.test_data))"""


#-----------------Pop
pop = Pop()
spop= SessionPop()
knn = ItemKNN()

model_list=[pop, spop, knn]
for model in model_list:
    model.fit(train_data)
    res=evaluate_sessions(model, test_data, train_data, cut_off=5)
    print('Recall@{}: {}'.format(k,res[0]))
    print('MRR@{}: {}'.format(k,res[1]))

