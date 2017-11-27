from textblob import TextBlob
from collections import Counter
import re
from nltk import ngrams
import nltk
from nltk.tokenize import TweetTokenizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim

def get_all_files_list(DIR):
	FILES = os.listdir(DIR)
	dfs = []
	for FILE in FILES:
		print(FILE)
		try:
			df = pd.read_csv(DIR+FILE,engine='c',na_filter=False,warn_bad_lines=False,error_bad_lines=False)
		except:
			df = pd.read_csv(DIR+FILE,engine='python',na_filter=False,warn_bad_lines=False,error_bad_lines=False)

		dfs.append(df)
	return FILES,dfs

def get_lemmas(text,token_func=None):
	def to_lemma(word,pos):
	    if pos[0] == 'J':
	        pos = wn.ADJ
	    elif pos[0] == 'V':
	        pos = wn.VERB
	    elif pos[0] == 'N':
	        pos = wn.NOUN
	    elif pos[0] == 'R':
	        pos = wn.ADV
	    else:
	        pos = 'a'
	    return lemmatizer.lemmatize(word,pos=pos)
	tokens = token_func(text)
	tags = nltk.pos_tag(tokens)
	token = [to_lemma(x,pos=p) for x,p in tags]
	text_out = ' '.join(token)
	return text_out


def get_wsd(lemmatized_text):
	print(lemmatized_text)
	wsd_out = disambiguate(lemmatized_text,algorithm=maxsim, \
            context_is_lemmatized=False, similarity_option='wup', keepLemmas=False, prefersNone=True)
	return np.matrix(wsd_out)



def clean_sentence(text):
	if type(text) == str:
		text = re.sub(r"http\S+", "", text) #get rid of links
		text = re.sub("\\n|\\r|\\t", "", text) #get rid of \n
		#get rid off ReTweet
		p = re.compile('RT (\@\w+|\w+):? ')
		if type(p.search(text))!=type(None):
			text = text[p.search(text).end():]

		# attach all proper nouns together, in the form Monero Dark
		p = re.compile('[A-Z][\w\-\.]*( [A-Z][\w\-\.]*)+(?![A-Z])')
		nouns = []
		NNs = []
		tknzr = TweetTokenizer()
		cut = 0
		Ns = []
		while True:
			match = p.search(text[cut:])
			if type(match) == type(None):
				break
			word = match.group().replace(' ','-')
			Ns.append(word)
			NNs.append(Ns)
			start = match.start() + cut
			end = match.end() + cut
			text = text[:start]+word+text[end:]
			if cut < match.end()+cut:
				cut+=match.end()
			else:
				break
		#get rid off some punctuation
		no_string = string.punctuation+'’'+'“'+'“'+'‘'+'_'
		token = tknzr.tokenize(text) #keep emoji constructed by punctuation
		token = [x for x in token if x not in no_string]
		text = ' '.join(token)
	else:
		print('text type error: not string, clean sentence as none')
		text = ''
	return text



def get_tokens(text,n=1):
	if type(text) == str:
		text = re.sub('[0-9\']','',text)
		tknzr = TweetTokenizer()
		token = tknzr.tokenize(text)
	#    token = [PorterStemmer().stem(x) for x in token]
		token = list(ngrams(token,n))
	else:
		print('text type error: not string')
		token = ''
	return token



def train_model(texts,q=None,N=1,norm=True):
	grams = []
	for i in range(len(texts)):
		print(i)
		text = texts[i]
		n_gram = get_tokens(text,n=N)
		grams+=n_gram
	counts = Counter(grams)
	if norm==True:
		total_counts = len(grams)
		for gram in counts:
			counts[gram] /= float(total_counts)
	outputs = counts
	if q !=None:
		q.put(outputs)
	return outputs




def test_model_log(test_text,model=None,N=2):
	text = clean_sentence(text=test_text)
	log_prob = 0
	for token in get_tokens(text=text,n=N):
		p = np.log(model[token])
		if model[token]==0:
#                print('Unobserved testing data:',token,model[token])
			p = -100
		log_prob += p
	return log_prob

	outputs = []
	#P(l|s) = P(s|l) * P(l) / p(s)
	xs = tests
	if type(xs) != enumerate:
		enumerater = enumerate(xs)
	else:
		enumerater = xs
	orders=[]
	for i,x in enumerater:
		orders.append(i)
		text = x
		text = clean_sentence(text=text)
		log_prob = 0
		for token in get_tokens(text=text,n=N):
			p = np.log(model[token])
			if model[token]==0:
#                print('Unobserved testing data:',token,model[token])
				p = -100
			log_prob += p
		outputs.append(log_prob)
		print(i,log_prob)
	if q !=None:
		q.put([orders,outputs])
	return [orders,outputs]




def pos_tagger(text,token_func=None):
	tokens = token_func(text)
	tags = nltk.pos_tag(tokens)
	return tags
