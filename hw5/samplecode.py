import numpy as np
import string
from sklearn.metrics import f1_score
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU,LSTM
from keras.layers import TimeDistributed,Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import json
train_path = sys.argv[1]
test_path = sys.argv[2]
# output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 300
# nb_epoch = 1000
# batch_size = 128
nb_epoch = 1000
batch_size = 100
dense_num = 400
embedding_B = 840

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None :
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score_cus(y_true,y_pred):
    thresh = 0.3
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

    json_data = {
        'X_train' : X_train.tolist(),
        'Y_train' : Y_train.tolist(),
        'X_val' : X_val.tolist(),
        'Y_val' : Y_val.tolist(),
        'test_sequences' : test_sequences.tolist()
    }

    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
    embedding_dict = get_embedding_dict('glove.%dB.%dd.txt'% (embedding_B,embedding_dim))
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(128,activation='tanh',dropout=0.2,return_sequences=True))
    # model.add(GRU(256,activation='tanh',dropout=0.2,return_sequences=True))
    model.add(GRU(128,activation='tanh',dropout=0.2,return_sequences=False))

    model.add(Dense(dense_num*2,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dense_num,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(int(dense_num/2),activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(int(dense_num/4),activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score_cus])
   
    earlystopping = EarlyStopping(monitor='val_f1_score_cus', patience = 50, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath="%s_%s_%s.hdf5" % (str(embedding_dim),str(dense_num),str(batch_size)),
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score_cus',
                                 mode='max')
    
    hist = model.fit(X_train, Y_train, 
                     validation_data=(X_val, Y_val),
                     epochs=nb_epoch, 
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])

    model.load_weights("%s_%s_%s.hdf5" % (str(embedding_dim),str(dense_num),str(batch_size)))

    score = model.evaluate(X_val, Y_val)
    model.save("model/%s_%s_%s_%s.h5" % (str(score[1]),str(embedding_dim),str(dense_num),str(batch_size)))
    with open("%s_%s_%s_%s.json" % (str(score[1]),str(embedding_dim),str(dense_num),str(batch_size)), 'w') as f:
        json.dump(json_data, f)

    # Y_pred = model.predict(test_sequences)
    # thresh = 0.3
    # with open("ans_%s_%s_%s_%s.csv" % (str(score[1]),str(embedding_dim),str(dense_num),str(batch_size)),'w') as output:
    #     print ('\"id\",\"tags\"',file=output)
    #     Y_pred_thresh = (Y_pred > thresh).astype('int')
    #     for index,labels in enumerate(Y_pred_thresh):
    #         labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
    #         labels_original = ' '.join(labels)
    #         print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    # with open("ans_%s_%s_%s_%s.csv" % (str(score[1]),str(embedding_dim),str(dense_num),str(batch_size)),'w') as output:
    #     print ('\"id\",\"tags\"',file=output)
    #     Y_pred_maxone = Y_pred/(1/Y_pred.max())
    #     Y_pred_thresh = (Y_pred_maxone > thresh).astype('int')
    #     for index,labels in enumerate(Y_pred_thresh):
    #         labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
    #         labels_original = ' '.join(labels)
    #         print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    threshold_choice = np.arange(0.01,0.31,0.01,dtype="float32")
    val_pred = model.predict(X_val)

    scores = []
    best_threshold = np.zeros(val_pred.shape[1])
    for i in range(val_pred.shape[1]):
        val_prob = np.array(val_pred[:,i])
        for choice in threshold_choice:
            val_ans = [(prob>=choice).astype('int') for prob in val_prob]
            scores.append(f1_score(Y_val[:,i],val_ans,average="micro"))
        scores = np.array(scores)
        index = np.where(scores==scores.max())
        best_threshold[i] = threshold_choice[index[0][0]]
        scores = []

    Y_pred = model.predict(test_sequences)
    Y_pred_thresh = np.array([[(Y_pred[i,j]>=best_threshold[j]).astype('int') for j in range(Y_pred.shape[1])] for i in range(Y_pred.shape[0])])
    # thresh = 0.3
    # print(Y_pred_thresh)
    with open("ans_%s_%s_%s_%s.csv" % (str(score[1]),str(embedding_dim),str(dense_num),str(batch_size)),'w') as output:
        print ('\"id\",\"tags\"',file=output)
        # Y_pred_thresh_1 = (Y_pred > thresh).astype('int')
        # print(Y_pred_thresh_1)
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
