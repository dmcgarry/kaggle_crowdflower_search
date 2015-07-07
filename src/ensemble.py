#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python
"""
Blends multiple solutions together to get a final prediction.
"""

import argparse
import pandas as pd
from os import listdir
from numpy import zeros, mean, var
from helperFunctions import convert_reg, quadratic_weighted_kappa, roundCut
from sklearn.cross_validation import train_test_split

##################
### Read Input ###
##################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cut", dest="cut", type=float,default=0.5)
parser.add_argument("--var", dest="variance", type=float,default=1)
parser.add_argument("--lower", dest="lower", type=float,default=2)
parser.add_argument("--upper", dest="upper", type=float,default=3)

args = parser.parse_args()

############
### Main ###
############

train = pd.read_csv("../data/train.csv")
train_train, valid_train, train_target, valid_target = train_test_split(train,train.median_relevance,test_size=0.20,random_state=2)
test_set = pd.read_csv("../data/test.csv")

files = [x for x in listdir("../data/") if x.startswith("pred_")]
valid = zeros((2032,len(files)),dtype="float")
test = zeros((22513,len(files)),dtype="float")
for i in range(len(files)):
	valid[:,i] = pd.read_csv("../data/"+files[i].replace('pred','valid'))['pred'].values
	test[:,i] = pd.read_csv("../data/"+files[i])['prediction'].values


valid_train['var'] = var(valid,axis=1)
valid_train['pred_raw'] = mean(valid,axis=1)
valid_train['pred'] = valid_train['pred_raw'].copy()
tmp = valid_train['var'] > args.variance
valid_train['pred'][tmp] = valid_train[tmp].pred_raw.apply(lambda x: 1 if x < args.lower else 4 if x > args.upper else x).values
valid_train['pred'] = roundCut(valid_train['pred'],args.cut)
print "Mean:", quadratic_weighted_kappa(valid_train.pred.astype(int),valid_target)
	
test_set['var'] = var(test,axis=1)
test_set['pred_raw'] = mean(test,axis=1)
test_set['pred'] = test_set['pred_raw'].copy()
tmp = test_set['var'] > args.variance
test_set['pred'][tmp] = test_set[tmp].pred_raw.apply(lambda x: 1 if x < args.lower else 4 if x > args.upper else x).values
test_set['pred'] = roundCut(test_set['pred'],args.cut)

pd.DataFrame({'id':test_set.id,'prediction':test_set['pred'].astype(int)}).to_csv("../data/pred.csv",index=False)
