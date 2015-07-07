def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    import numpy as np
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def convert_reg(y_pred):
	from numpy import round
	y_pred = round(y_pred)
	def trunc(x):
		if x < 1:
			return 1
		elif x > 4:
			return 4
		else:
			return x
	return [trunc(x) for x in y_pred]

def kappa_reg(y,y_pred):
	from helperFunctions import quadratic_weighted_kappa, histogram, confusion_matrix,convert_reg
	return quadratic_weighted_kappa(y,convert_reg(y_pred))

def cleanText(dat,removeHTML=True,removePunc=False,removeDigits=False,stemming=None):
	from re import sub
	if removeHTML:
		from bs4 import BeautifulSoup
		dat = [sub("\s+"," ",BeautifulSoup(x).get_text()).encode("ascii","replace") for x in dat]
	else:
		dat = [sub("\s+"," ",x) for x in dat]
	if removePunc:
		from string import punctuation
		dat = [''.join(ch for ch in s if ch not in set(punctuation)) for s in dat]
	if removeDigits:
		dat = [''.join(ch for ch in s if ch not in ['0','1','2','3','4','5','6','7','8','9']) for s in dat]
	if stemming == 'porter':
		from helperFunctions import PorterTokenizer
		dat = [' '.join(PorterTokenizer(x)) for x in dat]
	elif stemming == 'snowball':
		from helperFunctions import SnowballTokenizer
		dat = [' '.join(SnowballTokenizer(x)) for x in dat]
	elif stemming == 'lancaster':
		from helperFunctions import LancasterTokenizer
		dat = [' '.join(LancasterTokenizer(x)) for x in dat]
	return [sub('\W*\b\w{1,1}\b','',x) for x in dat]


def applyWeights(predList,weights):
	from numpy import zeros
	pred = zeros(predList[predList.keys()[0]].shape)
	for key in predList:
		pred += predList[key]*weights[key]
	return pred

def loadData(fpath="../data/",k=5,useDesc=True,labelFields=False,removeHTML=True,removePunc=False,removeDigits=False,stemming=None,SEED=55,load2vec=True):
	import pandas as pd
	from os import listdir
	from sklearn.cross_validation import KFold
	from helperFunctions import cleanText
	train = pd.read_csv(fpath+"train.csv").fillna('')
	cv = KFold(len(train), n_folds=k, shuffle=True, random_state=SEED)
	train['title'] = train.product_title
	del train['product_title']
	train['description'] = train.product_description
	del train['product_description']
	test = pd.read_csv(fpath+"test.csv").fillna('')
	test['title'] = test.product_title
	del test['product_title']
	test['description'] = test.product_description
	del test['product_description']
	#add extra vars
	train['query_words'] = [len(x.split(' ')) for x in train['query']]
	test['query_words'] = [len(x.split(' ')) for x in test['query']]
	train['title_len'] = [len(x) for x in train['title']]
	test['title_len'] = [len(x) for x in test['title']]
	train['desc_len'] = [len(x) for x in train['description']]
	test['desc_len'] = [len(x) for x in test['description']]
	#combine text fields
	if useDesc:
		vars = ['query','title','description']
	else:
		vars = ['query','title']
	for v in vars:
		train[v] = cleanText(train[v],removeHTML=removeHTML,removePunc=removePunc,removeDigits=removeDigits,stemming=stemming)
		test[v] = cleanText(test[v],removeHTML=removeHTML,removePunc=removePunc,removeDigits=removeDigits,stemming=stemming)	
	if labelFields:
		train['text'] = train.apply(lambda x: ' '.join([' '.join([v[0]+y for y in x[v].split(' ')]) for v in vars]),1)
		test['text'] = test.apply(lambda x: ' '.join([' '.join([v[0]+y for y in x[v].split(' ')]) for v in vars]),1)
	else:
		train['text'] = train.apply(lambda x: ' '.join([x[v] for v in vars]),1)
		test['text'] = test.apply(lambda x: ' '.join([x[v] for v in vars]),1)
	train['query_percent_title'] = train[['query','title']].apply(lambda row: 1.0*sum([q in row.title.lower().split(" ") for q in row['query'].split(" ")])/len(row['query'].split(" ")) ,1)
	test['query_percent_title'] = test[['query','title']].apply(lambda row: 1.0*sum([q in row.title.lower().split(" ") for q in row['query'].split(" ")])/len(row['query'].split(" ")) ,1)
	train['query_word_score'] = train.query_words*train.query_percent_title
	test['query_word_score'] = test.query_words*test.query_percent_title
	if load2vec:
		train_doc2vec = {}
		test_doc2vec = {}
		for f in  [x for x in listdir("../data/") if x.startswith("train_doc2vec")]:
			train_doc2vec[f] = pd.read_csv("../data/"+f)['doc2vec'].values
			test_doc2vec[f] = pd.read_csv("../data/"+f.replace('train','test'))['doc2vec'].values
		train['doc2vec'] = applyWeights(train_doc2vec,dict([(k,1.0/len(train_doc2vec)) for k in train_doc2vec.keys()]))
		test['doc2vec'] = applyWeights(test_doc2vec,dict([(k,1.0/len(test_doc2vec)) for k in test_doc2vec.keys()]))
		train_word2vec = {}
		test_word2vec = {}
		for f in  [x for x in listdir("../data/") if x.startswith("train_word2vec")]:
			train_word2vec[f] = pd.read_csv("../data/"+f)['word2vec'].values
			test_word2vec[f] = pd.read_csv("../data/"+f.replace('train','test'))['word2vec'].values
		train['word2vec'] = applyWeights(train_word2vec,dict([(k,1.0/len(train_word2vec)) for k in train_word2vec.keys()]))
		test['word2vec'] = applyWeights(test_word2vec,dict([(k,1.0/len(test_word2vec)) for k in test_word2vec.keys()]))
	return train.median_relevance, train, test, cv

def SnowballTokenizer(s):
	from nltk import word_tokenize          
	from nltk.stem import SnowballStemmer
	stemmer = SnowballStemmer('english')
	return [stemmer.stem(t) for t in word_tokenize(s)]

def LancasterTokenizer(s):
	from nltk import word_tokenize          
	from nltk.stem import LancasterStemmer
	stemmer = LancasterStemmer()
	return [stemmer.stem(t) for t in word_tokenize(s)]

def PorterTokenizer(s):
	from nltk import word_tokenize          
	from nltk.stem import PorterStemmer 
	stemmer = PorterStemmer()
	return [stemmer.stem(t) for t in word_tokenize(s)]
	
def trainText(train,target,test,pipeline,grid,cv,n_jobs=10,reg=False):
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import make_scorer
	from helperFunctions import quadratic_weighted_kappa, histogram, confusion_matrix, kappa_reg
	from sklearn.cross_validation import train_test_split
	from copy import copy
	from pandas import DataFrame
	if reg:
		scorer = make_scorer(kappa_reg, greater_is_better = True)
	else:
		scorer = make_scorer(quadratic_weighted_kappa, greater_is_better = True)
	model = GridSearchCV(estimator = pipeline, param_grid=grid, scoring=scorer,verbose=10, n_jobs=n_jobs, iid=False, refit=True, cv=cv)
	model.fit(train,target)
	print "Kappa:  %0.3f" % model.best_score_
	print "Params:", model.best_params_
	train_train, valid_train, train_target, valid_target = train_test_split(train,target,test_size=0.20,random_state=2)
	train_train = DataFrame(train_train,columns=train.columns)
	valid_train = DataFrame(valid_train,columns=train.columns)
	valid_model = copy(model.best_estimator_)
	valid_model.fit(train_train,train_target)
	return DataFrame({'id':test.id,'prediction':model.predict(test)}), DataFrame({'target':valid_target,'pred':valid_model.predict(valid_train)})

def get_sim(model,vocab,s1,s2):
	try:
		s1 = [x for x in s1.split(' ') if x in vocab]
	except AttributeError:
		s1 = []
	try:
		s2 = [x for x in s2.split(' ') if x in vocab]
	except AttributeError:
		s2 = []
	if s1 and s2:
		return model.n_similarity(s1,s2)
	else:
		return 0
	
def get_sim_array(model,vocab,x,y):
	from helperFunctions import get_sim
	if len(y.shape) > 1:
		return [get_sim(model,vocab,x.ix[i],' '.join(y.ix[i])) for i in range(len(x))]
	else:
		return [get_sim(model,vocab,x.ix[i],y.ix[i]) for i in range(len(x))]

def trainDoc2Vec(X,y,train,test,n_jobs=-1,**kwargs):
	from gensim.models.doc2vec import Doc2Vec, LabeledSentence
	from joblib import Parallel, delayed 
	from helperFunctions import get_sim_array
	model = Doc2Vec()
	for arg in kwargs:
		setattr(model,arg,kwargs[arg])
	model.build_vocab([LabeledSentence(X.ix[i].split(' '),y.ix[i].split(' ')) for i in range(len(X))])
	vocab = model.vocab.keys()
	return Parallel(n_jobs=n_jobs)(delayed(get_sim_array)(model,vocab,x[0],x[1]) for x in [(train['query'],train['title']),(test['query'],test['title'])])

def trainWord2Vec(fpath,train,test,n_jobs=-1,**kwargs):
	from gensim.models.word2vec import Word2Vec, LineSentence
	from joblib import Parallel, delayed 
	from helperFunctions import get_sim_array
	model = Word2Vec()
	for arg in kwargs:
		setattr(model,arg,kwargs[arg])
	model.build_vocab(LineSentence(fpath))
	vocab = model.vocab.keys()
	return Parallel(n_jobs=n_jobs)(delayed(get_sim_array)(model,vocab,x[0],x[1]) for x in [(train['query'],train['title']),(test['query'],test['title'])])

def roundCut(x,cut=0.5):
	def f(i,cut):
		if i - int(i) >= cut:
			x = int(i) + 1
		else:
			x = int(i)
		if x > 4:
			return 4
		elif x < 1:
			return 1
		else:
			return x
	return [f(i,cut) for i in x]		
	