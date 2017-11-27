import datetime
import re
import pandas as pd
import numpy as np
import nltk
import nltk.sentiment.vader as vd
import tweepy
from tweepy import OAuthHandler
from collections import Counter
from functools import reduce
import time
import json
import logging
import os
import sys
try:
	CWDIR = os.path.abspath(os.path.dirname(__file__))
except:
	CWDIR = os.getcwd()

#sys.path.append(CWDIR+'/lib/utils')

output_directory = CWDIR+'/../../output/'
pd.options.mode.chained_assignment = None  # default='warn'


os.system('mkdir -p {}'.format('intermediate/'))
os.system('mkdir -p {}/logs'.format('intermediate/'))
os.system('mkdir -p {}/tmp'.format('intermediate/'))

os.system('mkdir -p {}'.format('output/'))
os.system('mkdir -p {}/Ttrends'.format('output/'))
os.system('mkdir -p {}/Gtrends'.format('output/'))
os.system('mkdir -p {}/good_tweets'.format('output/'))
os.system('mkdir -p {}/full_tweets'.format('output/'))

logging.basicConfig(filename='intermediate/logs/scrapetweets.log',level=logging.WARNING)
#logging.basicConfig(level=logging.ERROR)
#logging.basicConfig(level=logging.WARNING)
#logging.basicConfig(level=logging.INFO)
lgr = logging.getLogger('tweets_utils.py')
pd.options.mode.chained_assignment = 'warn'  #stop thethe chain assignment copy warning


def get_all_files_list(DIR):
	import os
	files = os.listdir(DIR)
	files = [x for x in files if type(re.search('^(?:\.)',x)) == type(None)]
	dfs = []
	for file_i in files:
		try:
			df = pd.read_csv(DIR+file_i,engine='c',na_filter=False,warn_bad_lines=False,error_bad_lines=False)
		except:
			try:
				df = pd.read_csv(DIR+file_i,engine='python',na_filter=False,warn_bad_lines=False,error_bad_lines=False)
			except Exception as error:
				print('Failed to read {}, the error message is:'.format(file_i))
				print(error)

		dfs.append(df)
	return files,dfs

def setup_tweets_search(account_id=0):
	"""
	consumer_key = 'ZG6SSDvT52qF2Dht9wDSQ5XmX'
	consumer_secret = 'VLWH9NMBgBu7LJe2f1H24roj6y5d9GZg2P5ewDBtnVNhzHkd6L'
	access_token = '3820952056-6oEDsEdZH2nwOE9HFSHVRDd7vBiaYE2gRmV7GjL'
	access_secret = 'oPbj1639lYoifUCJMFWCWr6pedKDbxCBFPC70Hj1vKTve'
	"""
	df=pd.read_csv(CWDIR+'/../../input/twitter_accounts.csv')
	consumer_key = df['consumer_key'].iloc[account_id]
	consumer_secret = df['consumer_secret'].iloc[account_id]
	access_token = df['access_token'].iloc[account_id]
	access_secret = df['access_secret'].iloc[account_id]
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)

	#api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
	api = tweepy.API(auth)
	n_row = df.shape[0]
	return(api,n_row)

def word_in_text(word, text):
	word = word.lower()
	text = text.lower()
	match = re.search(word, text)
	if match:
		return True
	return False


def process_text(t):
	tokenizer = nltk.TweetTokenizer()
	tkn = tokenizer.tokenize(t)
	p = re.compile('^((?!http).)\w+$')
	r = list(map(lambda x: p.match(x) ,tkn))
	r = [x.group() for x in r if x != None]
	return ' '.join(r)

"""
time_length=time_length
n_tweets=n_tweets
account_id=account_id
cut_time_low = past_time_utc
tweets_sampling=tweets_sampling
e_i=e_i
"""


def get_tweets(term="bitcoin",time_length=[0,0,30,0],n_tweets=100000,account_id=0,cut_time_low=datetime.datetime.utcnow()-datetime.timedelta(days=0,hours=0,minutes=30,seconds=0),tweets_sampling=False,e_i=0):
	file_name ='intermediate/tmp/senti_df_'+term+'.csv'
	if re.search('[\"\']',term) == type(None):
		search_term = '\"'+term+'\"'
	else:
		search_term = term
	error = NameError
	output = []
	#calculate the gap between query time_length with today
	n_day,n_hour,n_min,n_sec = time_length
	time_gap = datetime.timedelta(days=n_day,hours=n_hour,minutes=n_min,seconds=n_sec)
	#get past_time
	now = datetime.datetime.now().replace(microsecond=0)
	past_time= now-time_gap
#	lgr.info('*******************************')
	lgr.info('Start searching tweets for {}'.format(search_term))
	lgr.info('Will stop when {} reached, or {} tweets collected'.format(str(past_time),n_tweets))
#	lgr.info('*******************************')
	#get day gap
	day_gap = time_gap.days
	today = datetime.date.today()
	since_day = today-datetime.timedelta(days=day_gap+2)
	n_row = 20
#	e_i = 0
	min_id = 0
	#get the min_id
	while min_id ==0:
		try:
			tweets = pd.DataFrame(columns=[#'compound', 'neg', 'neu', 'pos',
					'retweet_count', 'tweet_favorite_count', 'tweet_created_at', 'tweet_text', 'tweet_id',
				   'author', 'author_created_at', 'author_following_count',
				   'author_follower_count', 'author_posts_count', 'author_favorite_count','author_listed_count','author_id',
				   'author_location', 'user', 'user_created_at','user_age_days', 'user_following_count',
				   'user_follower_count', 'user_posts_count', 'user_favorite_count','user_listed_count','user_id',
				   'user_location','is_spam'])
			tweets.to_csv(file_name,header=True,index=False)

			api, n_row = setup_tweets_search(account_id=account_id)
			r0 = api.search(q=search_term, lang='en',since=str(since_day),count=1)
			if len(r0) > 0:
				min_id = r0[0].id
			else:
				lgr.error('Failed to find any tweets for {} between {} and {}'.format(search_term,str(today),str(since_day)))
				return output,account_id
			e_i = 0
		except Exception as error:
#			lgr.warning('*******************************')
			lgr.warning("Failed at {} with account {}".format(search_term,account_id))
#			if str(error.args).find('88') != -1: #then we get error 88
			lgr.warning(error)
#            failed_terms.append(search_term)
			account_id += 1
			e_i+=1
			if account_id < n_row:
				api, n_row = setup_tweets_search(account_id=account_id)
			else:
				account_id = 0
			lgr.warning('Changing account number to {}'.format(account_id))
	#		lgr.warning('*******************************')

			if e_i >=n_row:
		#		lgr.warning('*******************************')
				print('Running out of account_ids, now sleep for 300 seconds...')
			#	lgr.warning('*******************************')
				time.sleep(300) #
				e_i= 0



	#start querying
	iter_time = int(np.ceil(n_tweets/100)) #iter_time is used to limit tweets queries by returning tweeets number
	results = []
	late_time = now
	i=0
	count = 0
	jump_rate=1
	dur = 0.

	while i< iter_time:
		try:
			late_time0 = late_time
			one_piece = datetime.timedelta(days=1).total_seconds()/24/4 #we divide a day to 100 pieces (15 mins/piece)
			dur0 = dur
			min_id0 = min_id
			if dur0 < one_piece:
				jump_rate += 0.8
			elif dur0>=one_piece:
				jump_rate -= 0.3
			if jump_rate <0:
				jump_rate = 0
			if dur0 == 0: # for the case where twitter return 1 tweet repeatedly
				jump_rate = 0

			if tweets_sampling==False:
				jump_rate = 0
			result = api.search(q=search_term, lang='en',max_id=min_id,since=str(since_day),count=100)
			result_length = len(result)
#			output+=result
			tweets = get_tweets_df(tweets_data=result,word_list=[])
			#get a file
			#get a row number
			tweets.to_csv(file_name,header=False,mode='a',index=False)
#			del locals()['tweets','result']
			# if 0 result retrieved, then program stops
			if result_length==0:
				lgr.warning('Searching for {} stops at {}th batch.'.format(search_term,i))
				return output,account_id

			early_time = min([r.created_at for r in result])
			late_time = max([r.created_at for r in result])
			dur = (late_time0 - late_time).total_seconds()
			min_id = min([r.id for r in result])
			max_id = max([r.id for r in result])
			id_gap = max_id - min_id
			if dur ==0:
				id_jump = 0
			else:
				id_jump = int(id_gap*one_piece/dur) #get the estimate of the density
			#for the next iteration
			min_id = int(min_id - id_jump*jump_rate)
			if (min_id0 == min_id) and (result_length==1):
				lgr.warning('Searching for {} stops, {} tweets collected.'.format(search_term,len(output)))
				lgr.warning('Earliest tweet Twitter can give is after {}'.format(str(early_time)))
				lgr.warning('  rather than what you originally asked: {}'.format(str(past_time)))
				break
#				return output,account_id

			lgr.info('     {} {}: {} tweets found between {} and {}'.format(search_term,i,result_length,early_time,late_time))
			lgr.info('     Latest tweet id {} id_jump ={} jump_rate = {}'.format(min_id,id_jump,str(jump_rate)[:5]))
			if early_time <= cut_time_low:
				lgr.info('Successfully collect tweets for {}, scraper stops!'.format(str(time_gap)))
				break
			e_i = 0 #reset error_iteration times when successfully getting one result
			i+=1
		except Exception as error:
			print(error)
			lgr.warning("Failed at {} with account, {}".format(search_term,account_id))
			#change account when getting error 88
#			if str(error.args).find('88') != -1:
			account_id += 1
			e_i+=1
			if account_id < n_row:
				api, n_row = setup_tweets_search(account_id=account_id)
			else:
				account_id = 0
			lgr.warning('Changing account number to {}'.format(account_id))
			# try
			if e_i >=n_row:
				print('Running out of account_ids, now sleep for 300 seconds...')
				time.sleep(300) #
				e_i = 0


	lgr.info('Get {} tweets from'.format(len(output)))
	lgr.info('        {} to'.format(str(now)))
	lgr.info('        {}'.format(str(past_time)))



	return output,account_id



def get_tweets_df(tweets_data,word_list=[]):
	if len(tweets_data) ==0:
		tweets = pd.DataFrame(columns=[#'compound', 'neg', 'neu', 'pos', 'retweet_count',
			   'tweet_favorite_count', 'tweet_created_at', 'tweet_text', 'tweet_id',
			   'author', 'author_created_at', 'author_following_count',
			   'author_follower_count', 'author_posts_count', 'author_favorite_count','author_listed_count','author_id',
			   'author_location', 'user', 'user_created_at','user_age_days', 'user_following_count',
			   'user_follower_count', 'user_posts_count', 'user_favorite_count','user_listed_count','user_id',
			   'user_location','is_spam'])
		return tweets

	tweets = pd.DataFrame()
	tweets['retweet_count'] = list(map(lambda tweet: tweet.retweet_count, tweets_data))
	tweets['tweet_favorite_count'] = list(map(lambda tweet: tweet.favorite_count, tweets_data))
	tweets['tweet_created_at'] = list(map(lambda tweet: tweet.created_at, tweets_data))
	tweets['tweet_text'] = list(map(lambda tweet: tweet.text, tweets_data))
	tweets['tweet_id'] = list(map(lambda tweet: tweet.id_str, tweets_data))
	tweets['author'] = list(map(lambda tweet: tweet.author.screen_name, tweets_data))
	tweets['author_created_at'] = list(map(lambda tweet: tweet.author.created_at, tweets_data))
	tweets['author_following_count'] = list(map(lambda tweet: tweet.author.friends_count, tweets_data))
	tweets['author_follower_count'] = list(map(lambda tweet: tweet.author.followers_count, tweets_data))
	tweets['author_posts_count'] = list(map(lambda tweet: tweet.author.statuses_count, tweets_data))
	tweets['author_favorite_count'] = list(map(lambda tweet: tweet.author.favourites_count, tweets_data))
	tweets['author_listed_count'] = list(map(lambda tweet: tweet.author.listed_count, tweets_data))
	tweets['author_id'] = list(map(lambda tweet: tweet.author.id_str, tweets_data))
	tweets['author_location'] = list(map(lambda tweet: tweet.author.location if tweet.user.location != None else None, tweets_data))

	tweets['user'] = list(map(lambda tweet: tweet.user.screen_name, tweets_data))
	tweets['user_created_at'] = list(map(lambda tweet: tweet.user.created_at, tweets_data))
	tweets['user_age_days'] = np.array([x.days for x in tweets.tweet_created_at - tweets.user_created_at])
	tweets['user_following_count'] = list(map(lambda tweet: tweet.user.friends_count, tweets_data))
	tweets['user_follower_count'] = list(map(lambda tweet: tweet.user.followers_count, tweets_data))
	tweets['user_posts_count'] = list(map(lambda tweet: tweet.user.statuses_count, tweets_data))
	tweets['user_favorite_count'] = list(map(lambda tweet: tweet.user.favourites_count, tweets_data))
	tweets['user_listed_count'] = list(map(lambda tweet: tweet.user.listed_count, tweets_data))
	tweets['user_id'] = list(map(lambda tweet: tweet.user.id_str, tweets_data))
	tweets['user_location'] = list(map(lambda tweet: tweet.user.location if tweet.user.location != None else None, tweets_data))
	tweets['is_spam'] = None

	#get rid of repeated tweets
	repeated_tweets_ids = [k for k,x in list(Counter(tweets.tweet_id).items()) if x >1]
	judge=[]
	for i in range(tweets.shape[0]):
		one_id = tweets.tweet_id.iloc[i]
		if one_id not in repeated_tweets_ids:
			judge.append(True)
		else:
			judge.append(False)
			repeated_tweets_ids = [x for x in repeated_tweets_ids if x != one_id]
	tweets = tweets.loc[judge,:]

	# get the key_word list columns from the text, there's no value as default
	if len(word_list)!=0:
		for key in word_list:
			tweets['tweet_text'] = list(tweets['tweet_text'].apply(lambda tweet: process_text(tweet)))
			tweets[key] = list(tweets['tweet_text'].apply(lambda tweet: word_in_text(key,tweet)))

	return tweets

def read_tweets_csv(file_loc):
	df = pd.DataFrame(columns=['compound', 'neg', 'neu', 'pos', 'retweet_count',
		       'tweet_favorite_count', 'tweet_created_at', 'tweet_text', 'tweet_id',
		       'author', 'author_created_at', 'author_following_count',
		       'author_follower_count', 'author_posts_count', 'author_favorite_count','author_listed_count','author_id',
		       'author_location', 'user', 'user_created_at','user_age_days', 'user_following_count',
		       'user_follower_count', 'user_posts_count', 'user_favorite_count','user_listed_count','user_id',
		       'user_location','is_spam'])
	datatypes = {'author': str,
		 'author_created_at': str,
		 'author_favorite_count': int,
		 'author_follower_count': int,
		 'author_following_count': int,
		 'author_id': str,
		 'author_listed_count': int,
		 'author_location': str,
		 'author_posts_count': int,
		 'is_spam': str,
		 'retweet_count': int,
		 'tweet_created_at': str,
		 'tweet_favorite_count': int,
		 'tweet_id': str,
		 'tweet_text': str,
		 'user': str,
		 'user_age_days': int,
		 'user_created_at': str,
		 'user_favorite_count': int,
		 'user_follower_count': int,
		 'user_following_count': int,
		 'user_id': str,
		 'user_listed_count': int,
		 'user_location': str,
		 'user_posts_count': int}
	try:
		#dtype={"user_id": int, "username": object},
		df = pd.read_csv(file_loc,engine='c',na_filter=False,warn_bad_lines=False,error_bad_lines=False,dtype=datatypes)
	except:
		try:
			df = pd.read_csv(file_loc,engine='python',na_filter=False,warn_bad_lines=False,error_bad_lines=False, dtype= datatypes)
		except Exception as error:
			print('Failed to read {}, the error message is:'.format(file_loc))
			print(error)

	for i in range(df.shape[0]):
		try:
			df.loc[i,'tweet_created_at'] = datetime.datetime.strptime(df.loc[i,'tweet_created_at'],'%Y-%m-%d %H:%M:%S')
		except:
			df.loc[i,'tweet_created_at'] = df.loc[0,'tweet_created_at']
		try:
			df.loc[i,'author_created_at'] = datetime.datetime.strptime(df.loc[i,'author_created_at'],'%Y-%m-%d %H:%M:%S')
		except:
			df.loc[i,'author_created_at'] = df.loc[0,'user_created_at']
		try:
			df.loc[i,'user_created_at'] = datetime.datetime.strptime(df.loc[i,'user_created_at'],'%Y-%m-%d %H:%M:%S')
		except:
			df.loc[i,'user_created_at'] = df.loc[0,'user_created_at']

	df = df.sort_values('tweet_created_at')

	return df

def senti_scoring(tweets):
	if type(tweets) ==type(None):
		return tweets
	sia = vd.SentimentIntensityAnalyzer()
	if tweets.shape[0] != 0 :
		sentiments = list(map(lambda text: sia.polarity_scores(text), list(tweets['tweet_text'])))
		tweets = pd.concat([pd.DataFrame(sentiments),tweets],axis=1)
	else:
		tweets = pd.concat([pd.DataFrame(columns=['compound','pos','neu','neg']),tweets],axis=1)
	return tweets


def bin_judge_finder(df,bin_up,bin_gap):
	bin_low = bin_up - bin_gap
	judge = (df['tweet_created_at'] <= bin_up) & (df['tweet_created_at'] > bin_low)
	bin_up = bin_low
	return judge,bin_up

"""
senti_df=tweets_senti
n_pieces=n_pieces
cut_time_up=now_utc
cut_time_low=past_time_utc
update_spam_list=False
"""

def get_bin_dfs(senti_df,n_pieces,cut_time_up=None,cut_time_low=None,spam_list=[],update_spam_list=False):
#	print(senti_df.shape[0])

	if len(senti_df) ==0:
#    senti_score = pd.DataFrame(data=[[cut_time_up,0,1,0,1,0,0]])
		return 	pd.DataFrame(data=[[cut_time_up,0,0,1,0,1,-1,0]],columns=['bin_start_date','compound_score','twitter_trend_positive','volume_tweets_positive','twitter_trend_negative','volume_tweets_negative','number_tweets','number_tweets_rejected'])
#        senti_score.columns=['bin_start_date','compound_score','twitter_trend_positive','volume_tweets_positive','twitter_trend_negative','volume_tweets_negative','number_tweets','number_tweets_rejected']
	#	return 	senti_score
	senti_df['tweet_created_at'] = list(senti_df['tweet_created_at'].fillna(method='backfill'))
	try:
		senti_df['tweet_created_at']=[datetime.datetime.strptime(text,'%Y-%m-%d %H:%M:%S') for text in list(senti_df['tweet_created_at'])]
	except:
		pass

	if cut_time_low == None:
#        bin_low0 = min(senti_df['tweet_created_at']).replace(hour=0,minute=0,second=0)
		cut_time_low = min(list(senti_df['tweet_created_at']))

	if cut_time_up == None:
#        bin_low0 = min(senti_df['tweet_created_at']).replace(hour=0,minute=0,second=0)
		cut_time_up = max(list(senti_df['tweet_created_at']))

#    bin_up = cut_time_up.replace(minute=0,second=0) #if there's not enough for one hour
	hour = cut_time_up.hour

	if hour < 23:
		bin_up = cut_time_up.replace(hour=hour+1,minute=0,second=0)
	else:
		bin_up = cut_time_up.replace(hour=hour,minute=0,second=0)

	one_piece = datetime.timedelta(days=1).total_seconds()/n_pieces # to match google trend 1day we divide a day to 80 pieces            late_time0=late_time
	bin_gap = datetime.timedelta(seconds=one_piece)
	scores=[]
	score_back_up = [cut_time_up,0,0,1,0,1,-1,0]
	time_gap = cut_time_up-cut_time_low
	while bin_up > cut_time_low:
		#filter time
		bin_up0 = bin_up
		judge,bin_up = bin_judge_finder(senti_df, bin_up,bin_gap)
		senti_df_j = senti_df.loc[judge,:]


		score = get_senti_scores(senti_df_j=senti_df_j,cut_time_up=None,cut_time_low=None,fill_na=False,time_gap=time_gap,spam_list=spam_list,update_spam_list=update_spam_list)

		score[0] = bin_up0
		if score[-2] !=-1:
			score_back_up = score[:]
			score_back_up[3] = 0 #volume_pos
			score_back_up[5] = 0 #volume_neg
			score_back_up[6] = -1 #number_tweets
			score_back_up[7] = 0 #number_tweets_rejected
		elif score[-2] == -1:
			score = score_back_up
		scores.append(score)

	if len(scores)>0:
		output = pd.DataFrame(scores,columns=['bin_start_date','compound_score','twitter_trend_positive','volume_tweets_positive','twitter_trend_negative','volume_tweets_negative','number_tweets','number_tweets_rejected'])

	return(output)


#spam filter for tweets
def filter_tweets(senti_df,time_gap=datetime.timedelta(days=15),spam_list=[],white_list=[],update_spam_list=False):
	#a dynamic detection way
	if update_spam_list==False:
		return senti_df,senti_df,0,spam_list

	try:
		senti_df_gap = max(senti_df.tweet_created_at) - min(senti_df.tweet_created_at)
	except:
		senti_df_gap = datetime.timedelta(days=0)
	if time_gap.days !=0:
		total_bar = time_gap.days*3
	else:
		total_bar = 3
	bar = senti_df_gap/time_gap*total_bar
	if bar < 1:
		bar = 1
	repeated_users_ids = [k for k,x in list(Counter(senti_df.user_id).items()) if x >bar]
	repeated_users_ids_now = repeated_users_ids
	repeated_users_ids += spam_list
	#after assign spam to ids, update spam_list
	spam_list+=repeated_users_ids_now
	#update user list with spam_list, but if it's retweeted, don't treat it as spam, keep it in good tweets
	repeated_users_ids = [x for x in repeated_users_ids if x not in white_list]
	judge_user = [True if x not in repeated_users_ids else False for x in senti_df.user_id]
	judge_favorite = np.array([True if x > 5 else False for x in senti_df.tweet_favorite_count.astype(int)]).astype(int)
	judge_popular = np.array([True if x > 5 else False for x in senti_df.retweet_count.astype(int)]).astype(int)
	judges = np.bitwise_or(judge_favorite, judge_popular)
	judge_user = [True if (judge_user[i]==False) & (judges[i] == True) else judge_user[i] for i in range(len(judges))]

	#if a user's account created less than half year, then it might not be usual user
	safe_gap = 180
	judge_age =  senti_df.user_age_days > safe_gap

	# if a user post frequently in the past, he's a spammer
	user_avg_post_rate = senti_df['user_posts_count'].astype(float)/(senti_df['user_age_days']+1)
	user_avg_post_judge = user_avg_post_rate < 3

	judge = judge_user & judge_age & user_avg_post_judge & (senti_df['user_posts_count'].astype(int)>10) & \
		(senti_df['user_follower_count'].astype(int)>5) & \
		(senti_df['user_favorite_count'].astype(int)>1)

	senti_df_T = senti_df.loc[judge,:]
	senti_df_T.loc[:,'is_spam'] = [0]*senti_df_T.shape[0]

	judge_neg = ~judge
	senti_df.loc[:,'is_spam'] = judge_neg.astype(int)
	n_filtered = senti_df.shape[0] - senti_df_T.shape[0]
	return senti_df_T,senti_df,n_filtered,spam_list


def get_senti_scores(senti_df_j,cut_time_up=None,cut_time_low=None,fill_na=True,time_gap=datetime.timedelta(days=15),spam_list=[],update_spam_list=False):
	#empty data
	if senti_df_j.shape[0] ==0:
		return 	[cut_time_up,0,0,1,0,1,-1,0]

	if fill_na == True:
		senti_df_j['tweet_created_at'] = list(senti_df_j['tweet_created_at'].fillna(method='backfill'))
	try:
		senti_df_j['tweet_created_at']=[datetime.datetime.strptime(text,'%Y-%m-%d %H:%M:%S') for text in list(senti_df_j['tweet_created_at'])]
	except:
		pass

	#filter spam
	senti_df_j,senti_df_full,n_filtered,spam_list = filter_tweets(senti_df_j,time_gap=time_gap,spam_list=spam_list,update_spam_list=update_spam_list)
	#empty data
	if senti_df_j.shape[0] ==0:
		return 	[cut_time_up,0,0,1,0,1,-1,0]
	#filter time
	if cut_time_low == None:
		cut_time_low = min(list(senti_df_j['tweet_created_at']))
	if cut_time_up == None:
		cut_time_up = max(list(senti_df_j['tweet_created_at']))
	judge_ = (senti_df_j['tweet_created_at'] >= cut_time_low) & (senti_df_j['tweet_created_at'] <= cut_time_up)
	senti_df_j = senti_df_j.loc[judge_,:]

	#empty data
	if senti_df_j.shape[0] == 0:
		return [cut_time_up,0,0,1,0,1,-1,0]

	#calculate score
	p_pos=senti_df_j["compound"]>=0
	p_neg = senti_df_j["compound"]<0
	volume_pos = sum(p_pos) + senti_df_j.loc[p_pos,"retweet_count"].sum()+1
	volume_neg = sum(p_neg) + senti_df_j.loc[p_neg,"retweet_count"].sum()+1
	senti_pos_sum = (senti_df_j.loc[p_pos,"compound"] * (senti_df_j.loc[p_pos,"retweet_count"]+1)).sum()
	senti_neg_sum = (senti_df_j.loc[p_neg,"compound"] * (senti_df_j.loc[p_neg,"retweet_count"]+1)).sum()
	senti_pos = senti_pos_sum/volume_pos # plus 1 to avoid division by zero
	senti_neg = senti_neg_sum/volume_neg
	volume = int(senti_df_j["retweet_count"].sum()+senti_df_j.shape[0]) #number of senti_df_j + retweet_count
	try:
		compound_score = senti_pos_sum/(senti_pos_sum-senti_neg_sum)
	except Exception as e: #0-division error
		compound_score = 0

	#filter time bins that have small amount of tweets
#	if senti_df_j.shape[0]<10:
#		senti_score = [cut_time_up,compound_score,senti_pos,volume_pos,senti_neg,volume_neg,-1,n_filtered]
#	else:
	senti_score = [cut_time_up,compound_score,senti_pos,volume_pos,senti_neg,volume_neg,senti_df_j.shape[0],n_filtered]
	return senti_score


#get google trend by time
def trend_score_after(G_df,time_length):
	n_day,n_hour,n_min,n_sec = time_length
	now = datetime.datetime.now().replace(microsecond=0)
	time_gap = datetime.timedelta(days=n_day,hours=n_hour,minutes=n_min,seconds=n_sec)
	#get past_time
	past_time= now-time_gap
	time_stamps_judge = (G_df['bin_start_date'] > past_time).values

	time_stamps = [x for x in G_df['bin_start_date'] if x > past_time]
	MEAN = G_df.loc[time_stamps_judge,:].iloc[:,1].mean()
	STD = np.std(G_df.loc[time_stamps_judge].iloc[:,1])
	G_df = G_df.loc[time_stamps_judge,:]
	try:
		NOW = G_df.iloc[-1,1]
	except:
		NOW = np.nan # for empty G_df
	G_score = (NOW-MEAN)/STD
	return G_score




def unzip_outs(outs,now_utc,past_time_utc,mode='GT'):
	#get information from loops
	senti_dfs = {}
	output_ls = []
	failed_terms = []
	G_dfs = {}
	T_dfs = {}

	for i in range(len(outs)):
		output, failed,senti,T,G =outs[i]
		output= output[:2]+[now_utc-past_time_utc]+output[2:]
		output_ls.append(output)
		try:
			term = outs[i][0][0]
		except:
			failed_terms.append(failed)
			term = outs[i][1]
		if mode.find('T') != -1:
			senti_dfs.update({term:senti})
			T_dfs.update({term:T})
		if mode.find('G') != -1:
			G_dfs.update({term:G})

	return (senti_dfs,output_ls,failed_terms,G_dfs,T_dfs)

def save_to_csv(output_df,file_name,index,encoding,save_judge=True):
	if save_judge ==True:
		output_df.to_csv(file_name,index=index,encoding=encoding)
	return

def get_csv_log(output_ls,mode,now,past_time,ENCODING,SAVE_JUDGE,intermediate_GT=None,intermediate_senti=None):
	# add ID
	currency_df = pd.read_csv(CWDIR+'/../../input/currency.csv')
	dict_id = dict(zip(currency_df['name'],currency_df['id']))

	IDs = []
	for i in range(len(output_ls)):
		topic = output_ls[i][0] # topic name
		try:
			topic_id=dict_id[topic]
		except:
			topic_id=-1
		IDs.append(topic_id)

	if len(output_ls) == 0:
		return [None]
	#output dataframes when having mistakes
	if (mode.find('GT') != -1) | (mode.find('TG') != -1):
		output_df = pd.DataFrame(output_ls)
		output_df = pd.concat([pd.DataFrame(IDs),output_df],axis=1)
		output_df.columns = ['id','name','start_date','time_gap','twitter_trend_compound','twitter_trend_positive','volume_tweets_positive ','twitter_trend_negative','twitter_trend_negative','number_tweets','number_tweets_rejected','google_trend']
		save_to_csv(output_df=output_df,file_name=output_directory+'allGTtrendsMatrix_'+str(now)+'---'+str(past_time)+'.csv',index=False,encoding=ENCODING,save_judge=SAVE_JUDGE)

	elif mode.find('T') != -1:
		output_df = pd.DataFrame(output_ls)
		output_df = pd.concat([pd.DataFrame(IDs),output_df],axis=1)
		output_df.columns = ['id','name','start_date','time_gap','twitter_trend_compound','twitter_trend_positive','volume_tweets_positive ','twitter_trend_negative','twitter_trend_negative','number_tweets','number_tweets_rejected']
		save_to_csv(output_df=output_df,file_name=output_directory+'allTtrendsMatrix_'+str(now)+'---'+str(past_time)+'.csv',index=False, encoding='utf-8',save_judge=SAVE_JUDGE)

	elif mode.find('G') != -1:
		output_df = pd.DataFrame(output_ls)
		output_df = pd.concat([pd.DataFrame(IDs),output_df],axis=1)
		output_df.columns = ['id','name','start_date','time_gap','google_trend']
		save_to_csv(output_df=output_df,file_name=output_directory+'allGtrendsMatrix'+str(now)+'---'+str(past_time)+'.csv',index=False,encoding=ENCODING,save_judge=SAVE_JUDGE)

	return output_df

def json_to_df(json_txt,aim_keys=[]):
	if len(aim_keys) == 0:
		keys = list(json_txt.keys())
	else:
		keys= aim_keys
	for k in keys:
		json_txt[k]['contents'] = pd.read_json(json_txt[k]['contents'])
		json_txt[k]['failed_terms'] = json.loads(json_txt[k]['failed_terms'])
	return json_txt
