import argparse
import numpy as np
import pandas as pd
import keras
from dan_qa.custom_layers import AverageWords, WordDropout
#from preprocess import PreProcessor
from tweets_utils import get_all_files_list
from topics_utils import clean_sentence,train_model,test_model_log,pos_tagger,get_tokens
from multitask_utils import multi_work
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, roc_curve, auc

from keras.layers import Embedding, Dense, Input, BatchNormalization, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad, Adam, RMSprop,SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.losses import binary_crossentropy,mean_squared_error,kullback_leibler_divergence, categorical_crossentropy

import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




embedding_dim = EMBEDDING_DIM=300
num_hidden_layers = 3
num_hidden_units = 300
num_epochs = 100
batch_size = 100
dropout_rate = 0.2
word_dropout_rate = 0.3
activation = 'relu'
MAX_NB_WORDS = MAX_SEQUENCE_LENGTH = 140

parser = argparse.ArgumentParser()
parser.add_argument('-data', help='location of dataset', default='data/out_split.pk')
parser.add_argument('-We', help='location of word embeddings', default='data/glove.6B.300d.txt')
parser.add_argument('-model', help='model to run: nbow or dan', default='dan')
parser.add_argument('-wd', help='use word dropout or not', default='y')

args = vars(parser.parse_args())

"""
pp = PreProcessor(args['data'],args['We'])
pp.tokenize()
data, labels, data_val, labels_val = pp.make_data()

"""

def get_data(sample_num=1000,tokenizer=None):
	texts = []
	labels=[]
	with open('data.trainspam.txt','r') as f:
		rows = f.read()
		rows = rows.split('\n')
		for i in range(len(rows)):
			row_ = rows[i].split('__label__')
			try:
				label = row_[1]
				text = row_[0]
				texts.append(text)
				labels.append(label)
			except:
				print(row_)


	choice = np.array([x=='1' for x in labels])
	texts = np.array(texts)
	texts0 = list(texts[~choice])
	texts1 = list(texts[choice])


	n_min = min(len(texts0),len(texts1))
	data = texts0[:n_min] + texts1[:n_min]
	data = np.array(data)

	if tokenizer == None:
		tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		tokenizer.fit_on_texts(data)
	else:
		pass
	choice_index = np.random.choice(list(range(len(data))),size=sample_num,replace=False)
	sequences = tokenizer.texts_to_sequences(data[choice_index])
	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	labels = ['0']*n_min + ['1']*n_min
	labels = np.array(labels)[choice_index]
	labels = keras.utils.to_categorical(labels, num_classes=2)

	return data,labels, tokenizer


def get_embedding():

	model_fasttext = fasttext.load_model('embedding_models/model.bin')
	embeddings_index = {}
	for k in model_fasttext.words:
		embeddings_index[k] = model_fasttext[k]

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector

	return embedding_matrix


"""
texts = multi_work(thelist=list(enumerate(data)),func=get_tokens,arguments=[],iterable_input=False,scaling_number=4,on_disk=False)
texts = sum(texts,[])
texts = list(dict(sorted(texts)).values())
MAX_SEQUENCE_LENGTH = max([len(x) for x in texts])
MAX_NB_WORDS = MAX_SEQUENCE_LENGTH

"""
"""
	if args['We'] == "rand":
#        model.add(Embedding(len(pp.word_index) + 1,embedding_dim,input_length=pp.MAX_SEQUENCE_LENGTH,trainable=False))
		model.add(Embedding(len(model_fasttext.words) + 1,embedding_dim,input_length=MAX_SEQUENCE_LENGTH,trainable=False))
	else:
#        model.add(Embedding(len(pp.word_index)+1,embedding_dim,weights=[embedding_matrix],input_length=pp.MAX_SEQUENCE_LENGTH,trainable=False))
		model.add(Embedding(len(model_fasttext.words),embedding_dim,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False))
"""


#    embedding_matrix = pp.get_word_embedding_matrix(embedding_dim)
def create_model():
	model = Sequential()
	model.add(Embedding(len(word_index) + 1,
								EMBEDDING_DIM,
								weights=[embedding_matrix],
								input_shape=(MAX_SEQUENCE_LENGTH,),
								trainable=False))

	if args['wd'] == 'y':
		model.add(WordDropout(word_dropout_rate))
	model.add(AverageWords())


	if args['model'] == 'dan':
		for i in range(num_hidden_layers):
			model.add(Dense(num_hidden_units))
			model.add(BatchNormalization())
			model.add(Activation(activation))
			model.add(Dropout(dropout_rate))

#    model.add(Dense(labels.shape[1]))
	model.add(Dense(2))
	model.add(BatchNormalization())
	model.add(Dropout(dropout_rate))
	model.add(Activation('softmax'))

	return model

"""
def load_trained_model(model_file_path=model_file_path,X_test=X_test,Y_test=Y_test,model_number = model_number):
    i = 0
    predScores = np.zeros((model_number+1, len(Y_test)))
    model_file_path1 = model_file_path+'_'+str(i)+'.h5'

    while True:
        model = create_model()
        model_file_path1 = model_file_path+'_'+str(i)+'.h5'

        model.load_weights(model_file_path1)
        modLoss = [1]

        if ~np.isnan(modLoss[-1]):
            print(i)
            predScores[i,:] = np.squeeze(model.predict(X_test))
            i += 1
            if i >model_number:
               break

    return predScores, model


predScores, model = load_trained_model()
"""

optimizer=Adam(lr=0.1, beta_1=0.9 ,decay=0.01)
optimizer = SGD(lr=0.1, decay=0.01, momentum=0.9, nesterov=True)
optimizer = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.05)

grids = [
		[Adam(lr=0.1, beta_1=0.9 ,decay=0.01),200],#good
		[Adam(lr=0.1, beta_1=0.9 ,decay=0.01),600],
		[Adam(lr=0.05, beta_1=0.9 ,decay=0.01),200],
		[Adam(lr=0.05, beta_1=0.9 ,decay=0.01),600],
		[SGD(lr=0.1, decay=0.01, momentum=0.9, nesterov=True),200],
		[SGD(lr=0.1, decay=0.01, momentum=0.9, nesterov=True),600],
		[SGD(lr=0.05, decay=0.01, momentum=0.9, nesterov=True),200],#good
		[SGD(lr=0.05, decay=0.01, momentum=0.9, nesterov=True),600],#good
		[RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.05),200],
		[RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.05),600],
		[RMSprop(lr=0.05, rho=0.9, epsilon=1e-08, decay=0.05),200], #good
		[RMSprop(lr=0.05, rho=0.9, epsilon=1e-08, decay=0.05),600]
		]
grids = [[Adam(lr=0.1, beta_1=0.9 ,decay=0.01),200]]
if __name__ == "__main__":

	X_train,Y_train,tokenizer = get_data(sample_num=5000,tokenizer=None)
	X_test,Y_test,tokenizer = get_data(sample_num=20000,tokenizer=tokenizer)
	outs=[]
	for optimizer,batch_size in grids:
		print(optimizer,batch_size)
		word_index = tokenizer.word_index
		embedding_matrix = get_embedding()

		model = create_model()
#		model.summary()
		model.compile(loss='categorical_crossentropy',optimizer= optimizer, metrics=['categorical_accuracy'])
		earlyStopping = EarlyStopping(monitor='val_acc',min_delta = 0.0005, patience=15, verbose=0, mode='auto')
		csv_logger = CSVLogger('csv_log.csv',append=True)
		mod = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=200, verbose=2, callbacks=[earlyStopping,csv_logger], shuffle=True, validation_split=0.2)
	#    model.fit(X_train,Y_train,batch_size=500,epochs=100,verbose=2)

		# get_embedding_layer_output = K.function([model.layers[0].input],[model.layers[0].output])
		# el_output = np.mean(get_embedding_layer_output([data])[0],axis=1)
		# print el_output

		# get_average_word_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
		# print get_average_word_layer_output([data])[0]
	#    model.fit(data,labels,batch_size=batch_size,epochs=num_epochs,validation_data=(data_val,labels_val))
		y_true = Y_test[:,1]
		y_predict = model.predict_classes(X_test)
		X_test[y_true != y_predict]
		from sklearn.metrics import confusion_matrix
		mat = confusion_matrix(y_true,y_predict,labels=[0,1])
		outs.append(mat)

avg_outs = []
for x in outs:
	avg_outs.append((x[1,1]/x[1].sum()))
#	+x[0,0]/x[0].sum())/2)


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
new_sentences=[]
for sentence in X_test:
	new_sentence = [reverse_word_map[x] for x in sentence if x!=0]
	print(new_sentence)
	new_sentences.append(new_sentence)


"""
	y_predict = model.predict(X_test)
	y_true = Y_test
	mean_fpr, mean_tpr, mean_thresholds = roc_curve(y_true, y_predict, pos_label=1,drop_intermediate=False)
"""
