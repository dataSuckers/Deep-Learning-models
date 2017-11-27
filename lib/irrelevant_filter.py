import re
from nltk.tokenize import TweetTokenizer
import pandas as pd
import string
import numpy as np
from nltk.corpus import wordnet as wn

def clean_noun(text):
	if type(text) == str:
		text = re.sub(r"http\S+", "", text) #get rid of links
		text = re.sub("\\n|\\r|\\t", "", text) #get rid of \n
		#get rid off ReTweet
		p = re.compile('RT (\@\w+|\w+):? ')
		if type(p.search(text))!=type(None):
			text = text[p.search(text).end():]

		#get rid off some punctuation
		tknzr = TweetTokenizer()
		no_string = string.punctuation+'’'+'“'+'“'+'‘'+'_'+'...'
		token = tknzr.tokenize(text) #keep emoji constructed by punctuation
		token = [x for x in token if x not in no_string]
		text = ' '.join(token)
	else:
		print('text type error: not string, clean sentence as none')
		text = ''
	return text

def disambiguate_words(words):
    words = np.array(words)
    judges = []
    for word in words:
        judge_x = []
        for x in re.split(' |\W',word):
            one_judge = len(wn.synsets(x.lower()))!=0 # if word in phrase amubiguous
            judge_x.append(one_judge)
        judge = sum(judge_x) ==0 #if no word in the phrase ambiguous, it's not ambiguous
        judges.append(judge)

    good_words = words[judges]
    bad_words = words[~np.array(judges)]
    return good_words,bad_words

df = pd.read_csv('input/companies.csv')
#df = pd.read_csv('input/currency.csv')
words = df.name.values
#clean nouns

words = [clean_noun(x) for x in words]

good_words,bad_words = disambiguate_words(words)
del df

#good_words,bad_words = good_words[:3],bad_words[:3]
