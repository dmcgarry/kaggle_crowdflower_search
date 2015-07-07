#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python

import argparse
import json
import pandas as pd
from helperFunctions import trainWord2Vec, loadData, cleanText

##################
### Read Input ###
##################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", dest="fname", type=str)
parser.add_argument("--external", dest="external_path", type=str, default="../data/external.txt")
parser.add_argument("--path", dest="path", type=str,default="../data/")
parser.add_argument("--stem", dest="stemming", type=str,default="None")
parser.add_argument("--html", dest="removeHTML", action='store_true')
parser.add_argument("--punc", dest="removePunc", action='store_true')
parser.add_argument("--digits", dest="removeDigits", action='store_true')
parser.add_argument("--seed", dest="SEED", type=int,default=22)

args = parser.parse_args()

############
### Main ###
############

target, train, test, cv = loadData(k=2,useDesc=False,labelFields=False,removeHTML=args.removeHTML,removePunc=args.removePunc,removeDigits=args.removeDigits,stemming=args.stemming,SEED=args.SEED,load2vec=False)
train['query'] = [x.lower() for x in train['query']]


data = open(args.external_path,'rb').readlines()
data = cleanText(data,removeHTML=args.removeHTML,removePunc=args.removePunc,removeDigits=args.removeDigits,stemming=args.stemming)
data = [x.lower() for x in data]
open('../data/tmp.txt','wb').write('\n'.join(data))

word2vec = trainWord2Vec('../data/tmp.txt',train,test)

pd.DataFrame({'id':train.id,'word2vec':word2vec[0]}).to_csv("../data/train_word2vec_"+args.fname+".csv",index=False)
pd.DataFrame({'id':test.id,'word2vec':word2vec[1]}).to_csv("../data/test_word2vec_"+args.fname+".csv",index=False)
