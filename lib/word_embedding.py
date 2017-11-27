import gensim
from textblob import TextBlob
from collections import Counter
import re
from nltk import ngrams
import nltk
from nltk.tokenize import TweetTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
#from tweets_utils import get_all_files_list
#from topics_utils import clean_sentence,train_model,test_model_log,pos_tagger,get_tokens
from multitask_utils import multi_work
import os
import pandas as pd
import numpy as np
import fasttext

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import *
'****************************clean texts*********************************'
#get the data


"""
DIR = './intermediate/good_tweets/'
FILES, senti_dfs = get_all_files_list(DIR)
labels = [[FILES[i]]*senti_dfs[i].shape[0] for i in range(len(FILES))]
labels = sum(labels,[])
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
texts1 = list(dict(sorted(texts)).values())


DIR = './intermediate/spam_tweets/'
FILES, senti_dfs = get_all_files_list(DIR)
labels = [[FILES[i]]*senti_dfs[i].shape[0] for i in range(len(FILES))]
labels = sum(labels,[])
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
texts0 = list(dict(sorted(texts)).values())


texts0 = list(set(texts0))
texts1 = list(set(texts1))
texts = texts0+texts1
with open('data.txt','w') as f:
    for i in range(len(texts1)):
        f.write(texts1[i]+'\n')



with open('data.trainspam.txt','w') as f:
    for i in range(len(texts1)):
        f.write(texts1[i]+'__label__1'+'\n')
    for i in range(len(texts0)):
        f.write(texts0[i]+'__label__0'+'\n')




df = pd.read_excel('ICOrating Screen Scrape.xlsx')
texts = list(df.Description.dropna())

with open('data.txt','w') as f:
    for i in range(len(texts)):
        f.write(texts[i]+'\n')


import fasttext
model = fasttext.skipgram(input_file='data.txt', output='model',lr=0.1,dim=300,thread=4,silent=0,\
    encoding='utf-8',word_ngrams=3,ws=5,epoch=100,min_count=5,loss='ns',minn=3,maxn=6)



#model = fasttext.load_model('embedding_models/model.bin')

classifier = fasttext.supervised('data.trainspam.txt', 'model', label_prefix='__label__')
classifier.predict_proba(texts1[:20])
classifier.predict_proba(texts0[:20])

#get test data
with open('data.txt','r') as f:
    texts=f.read()
    texts = texts.split('\n')[:-1]


"""

#model = fasttext.load_model('wiki.simple/wiki.simple.bin')
def text_vectorizer(text, model):
    numw = 0
    sent_vec = np.zeros(len(model['a']))
    if text == '':
        return sent_vec
    tokens = text.lower().split()
    for i in range(len(tokens)):
        token = tokens[i]
        vec = model[token]
        sent_vec += vec
        numw+=1
    output=sent_vec/sent_vec.dot(sent_vec)

    return output















#model = fasttext.load_model('model.vec')

df = pd.read_excel('ICOrating Screen Scrape.xlsx')
texts = list(df.Description.replace(np.nan,''))
vecs = multi_work(thelist=list(enumerate(texts)),func=text_vectorizer,arguments=[[model]],iterable_input=False,scaling_number=4,on_disk=False)
vecs = sum(vecs,[])
vecs = sum(vecs,[])
vecs = list(dict(sorted(vecs)).values())


feat_cols = [ 'dim'+str(i) for i in range(len(vecs[0])) ]
data_df = pd.DataFrame(vecs,columns=feat_cols)

#DIR = './intermediate/good_tweets/'
#FILES, senti_dfs = get_all_files_list(DIR)
#labels = [[FILES[i]]*senti_dfs[i].shape[0] for i in range(len(FILES))]
#labels = sum(labels,[])

labels = df['Coin Score Sector'].astype(str).values

#labels = ''.join(texts).split(' ')[:472]
data_df['label'] = labels
rndperm = np.random.permutation(data_df.shape[0])


pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_df[feat_cols].values)

data_df['pca-one'] = pca_result[:,0]
data_df['pca-two'] = pca_result[:,1]
data_df['pca-three'] = pca_result[:,2]

chart = ggplot( data_df, aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")

chart


n_sne = 7000

tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(data_df.loc[rndperm[:n_sne],feat_cols].values)

data_df_tsne = data_df.loc[rndperm[:n_sne],:].copy()
data_df_tsne['x-tsne'] = tsne_results[:,0]
data_df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( data_df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.8) \
        + ggtitle("tSNE dimensions colored by digit")
chart


from sklearn.cluster import KMeans, MiniBatchKMeans
kmeans = KMeans(n_clusters=19, random_state=0,verbose=2,max_iter=300,n_jobs=3,algorithm='full').fit(vecs)
labels = kmeans.labels_

df = pd.read_excel('ICOrating Screen Scrape.xlsx')
df = pd.concat([pd.DataFrame(labels,columns=['kmeans_predict']),df],axis=1)
df.to_excel('kmeans_predicted.xlsx',index=False)


"""
#
#model = fasttext.cbow('data.txt', 'model', lr=0.1, dim=300)

from sklearn.cluster import KMeans
import numpy as np
#This function creates the classifier
#n_clusters is the number of clusters you want to use to classify your data

from gensim.models import doc2vec
from collections import namedtuple


# Transform data (you can add more data preprocessing steps)

docs = []
doc1= texts
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in enumerate(doc1):
    words = text.lower().split()
    tags = [i]
    docs.append(analyzedDocument(words, tags))

model1 = doc2vec.Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
model2 = Word2Vec(texts, size=100, window=5, min_count=5, workers=4)




#model = Doc2Vec(t, size=100, window=8, min_count=5, workers=4)



import gensim
# let X be a list of tokenized texts (i.e. list of lists of tokens)
model = gensim.models.Word2Vec(X, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

"""
