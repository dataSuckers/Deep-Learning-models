from textblob import TextBlob
from collections import Counter
import re
from nltk import ngrams
import nltk
from nltk.tokenize import TweetTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from topics_utils import clean_sentence,train_model,test_model_log,pos_tagger,get_tokens,get_all_files_list
from multitask_utils import multi_work
import os
import pandas as pd

'****************************clean texts*********************************'
#get the data





DIR = './../coinscore_test/output/good_tweets/'
FILES, senti_dfs = get_all_files_list(DIR)
tweets = pd.concat(list(senti_dfs))
#tweets = senti_dfs[0]
#tweets = senti_dfs[term]
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
#clean text
#CPU times: user 40 s, sys: 44 ms, total: 40 s
#Wall time: 40 s
#%time texts = [clean_sentence(x) for x in tweets.text]
texts = multi_work(thelist=list(enumerate(tweets.tweet_text.values)),func=clean_sentence,arguments=[],iterable_input=False,scaling_number=4,on_disk=False)
texts = sum(texts,[])
texts = sum(texts,[])
texts = list(dict(sorted(texts)).values())


'****************************train models*********************************'

#model training shouldn't be put together
#model1= multi_work(thelist=texts,func=train_model,arguments=[1,True])
#dict_whole = train_model(texts,N=2,norm=True)
dict_whole = train_model(texts,N=1,norm=True)
model = dict_whole
'****************************tag models*********************************'
#nltk pos tags
#CPU times: user 4min 35s, sys: 1.55 s, total: 4min 36s
#Wall time: 4min 37s
#%time tags_nltk = [nltk.pos_tag(tknzr.tokenize(text)) for text in texts]
tknzr = TweetTokenizer()
#if arguments contain a function, it should be double bracketed
tags = multi_work(thelist=list(enumerate(texts)),func=pos_tagger,arguments=[[tknzr.tokenize]],iterable_input=False)
outs = []
for x in tags:
	outs+=x
tags =outs
tags = list(dict(sorted(tags)).values())

#textblob tags
#CPU times: user 5min 58s, sys: 2.18 s, total: 6min
#Wall time: 6min 1s
#%time tags = [TextBlob(x).tags for x in texts]
'****************************train noun models*********************************'

#tag_list=['NN','NNS','NNP','NNPS']
tag_list=['NNP','NNPS']
noun_sentence = [' '.join([w  for w,t in one_tag if t in tag_list]) for one_tag in tags]
dict_noun = train_model(noun_sentence,N=1,norm=True)
model = dict_noun
'****************************test models*********************************'



#test
#CPU times: user 1min 51s, sys: 844 ms, total: 1min 52s
#Wall time: 1min 52s
tests = list(tweets.text.values)
#%time outs= test_model_log(tests=tests,model=dict_whole,N=1)
test_outs = multi_work(thelist=list(enumerate(tests)),func=test_model_log,arguments=[[model,1]],iterable_input=False)
test_outs = sum(test_outs,[])
test_outs = list(dict(sorted(test_outs)).values())


'****************************tfidf encoding*********************************'


tfidf = TfidfVectorizer(tokenizer=get_tokens,stop_words='english',analyzer='char',ngram_range=(3,7),smooth_idf=True)
%time tfs = tfidf.fit_transform(texts)
feature_names = tfidf.get_feature_names()
response = tfs
for col in response.nonzero()[1]:
	print(feature_names[col], ' - ', max(response[:, 135015][1]))
