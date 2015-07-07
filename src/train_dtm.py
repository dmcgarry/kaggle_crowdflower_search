#!/na/home/dmcgarry/envs/dmcgarry_2015_q1/bin/python
"""
Runs a tf-idf transformation on training data and builds a model on the DTM.
"""

import argparse
from helperFunctions import loadData, trainText

##################
### Read Input ###
##################

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", dest="fname", type=str)
parser.add_argument("--grid", dest="grid_path", type=str)
parser.add_argument("--path", dest="path", type=str,default="../data/")
parser.add_argument("--k", dest="k", type=int,default=5)
parser.add_argument("--stem", dest="stemming", type=str,default="None")
parser.add_argument("--desc", dest="useDesc", action='store_true')
parser.add_argument("--html", dest="removeHTML", action='store_true')
parser.add_argument("--label", dest="labelFields", action='store_true')
parser.add_argument("--punc", dest="removePunc", action='store_true')
parser.add_argument("--digits", dest="removeDigits", action='store_true')
parser.add_argument("--reg", dest="reg", action='store_true')
parser.add_argument("--seed", dest="SEED", type=int,default=22)
parser.add_argument("--jobs", dest="n_jobs", type=int,default=10)

args = parser.parse_args()

############
### Main ###
############

target, train, test, cv = loadData(k=args.k,useDesc=args.useDesc,labelFields=args.labelFields,removeHTML=args.removeHTML,removePunc=args.removePunc,removeDigits=args.removeDigits,stemming=args.stemming,SEED=args.SEED)

execfile(args.grid_path)	

test_pred, valid_pred = trainText(train,target,test,pipeline,grid,cv,args.n_jobs,args.reg)
	
test_pred.to_csv("../data/pred_"+args.fname+".csv",index=False)
valid_pred.to_csv("../data/valid_"+args.fname+".csv",index=False)
