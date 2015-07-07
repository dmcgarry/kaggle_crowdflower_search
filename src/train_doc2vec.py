#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python

import argparse
import json
import pandas as pd
from helperFunctions import trainDoc2Vec, loadData, cleanText

##################
### Read Input ###
##################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", dest="fname", type=str)
parser.add_argument("--external", dest="external_path", type=str, default="../data/external.json")
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

external = pd.DataFrame(json.load(open(args.external_path,'rb')))
del external['rank']
external['query'] = cleanText(external['query'],removeHTML=args.removeHTML,removePunc=args.removePunc,removeDigits=args.removeDigits,stemming=args.stemming)
external['title'] = cleanText(external['title'],removeHTML=args.removeHTML,removePunc=args.removePunc,removeDigits=args.removeDigits,stemming=args.stemming)
for v in ['query','title']:
	external[v] = [x.lower() for x in external[v]]
	train[v] = [x.lower() for x in train[v]]
	test[v] = [x.lower() for x in test[v]]

external = external.append(train[target == 4][['query','title']])
external = external.reset_index()

doc2vec = trainDoc2Vec(external['query'],external['title'],train,test)

pd.DataFrame({'id':train.id,'doc2vec':doc2vec[0]}).to_csv("../data/train_doc2vec_"+args.fname+".csv",index=False)
pd.DataFrame({'id':test.id,'doc2vec':doc2vec[1]}).to_csv("../data/test_doc2vec_"+args.fname+".csv",index=False)
