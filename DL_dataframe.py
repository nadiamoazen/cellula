

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten, TimeDistributed, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import metrics
from keras import backend
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

def mcor(y_test, y_pred):
    #matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
 
 
    y_pos = K.round(K.clip(y_test, 0, 1))
    y_neg = 1 - y_pos
 
 
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
 
 
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
 
 
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
 
    return numerator / (denominator + K.epsilon())



def precision(y_test, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_test, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_test, y_pred):
    def recall(y_test, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_test, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_test, y_pred)
    recall = recall(y_test, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

################## LOADING THE DATA

dataZ= pd.DataFrame()    
for i in range(1, 101):
    file_index= str(i).zfill(3)
    data = pd.read_csv("F{0}.txt".format(file_index), sep=" ", header=None)
    # print(data,"Z{0}.txt".format(file_index) )
    dataZ = pd.concat([dataZ, data], axis=0) #dataZ = pd.concat([dataZ, data], axis=0)
    
dataZ = dataZ.iloc[:, 0:1]

dz= pd.DataFrame()                               
for i in range(0,  len(dataZ)//2048):
    dz =pd.concat([dz,  pd.DataFrame(np.array(dataZ[i*2048:(i+1)*2048]).reshape(-1,2048 )) ], axis=0)
    
dz.insert(2048,"y",0,True)


################## Building desired dataframe

dataS= pd.DataFrame()    
    
for i in range(1, 101):
    file_index= str(i).zfill(3)
    data = pd.read_csv("S{0}.txt".format(file_index), sep=" ", header=None)
   # print(data,"Z{0}.txt".format(file_index) )
   ## data.insert(1,"y",0,True)
    dataS = pd.concat([dataS, data], axis=0) #dataZ = pd.concat([dataZ, data], axis=0)
    

dataS = dataS.iloc[:, 0:1]

ds= pd.DataFrame()                               
for i in range(0,  len(dataS)//2048):
    ds =pd.concat([ds,  pd.DataFrame(np.array(dataS[i*2048:(i+1)*2048]).reshape(-1,2048 )) ], axis=0)
    
ds.insert(2048,"y",1,True)

#DATASET== pd.DataFrame()
#pd.concat([dataZ, data], axis=0)

DATASET = pd.concat ([ds, dz])
DATASET=DATASET.iloc[np.random.permutation(len(DATASET))]
DATASET=DATASET.reset_index(drop=True)



DATASET.head()
tgt = DATASET.y
tgt.unique()
tgt

#cols = ESR.columns 
# df = df[df.line_race != 0]
#tgt[tgt>1]=0
ax = sn.countplot(tgt,label="Count")

DATASET.isnull().sum()
DATASET.info()
DATASET.describe()
X = DATASET.iloc[:,0:2048].values
X.shape

y = DATASET.iloc[:,2048].values
y 

plt.subplot(511)
plt.plot(X[1,:])
plt.title('Classes')
plt.ylabel('uV')
plt.subplot(512)
plt.plot(X[7,:])
plt.subplot(513)
plt.plot(X[12,:])
plt.subplot(514)
plt.plot(X[0,:])
plt.subplot(515)
plt.plot(X[2,:])
plt.xlabel('Samples')

from sklearn.preprocessing import StandardScaler
###scaler = StandardScaler()
###scaler.fit(X)
###x = scaler.transform(X)
from keras.utils import to_categorical
from keras.utils import np_utils
#y = to_categorical(y)

#y= tf.keras.utils.to_categorical(y, num_classes=5)
#y=np_utils.to_categorical(y, num_classes=5)
#y=np_utils.to_categorical(y, num_classes=5)




from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

print (y_train[:3])
# print (y_train) # [5 3 1 ... 1 5 5]
#print (y_train.shape) #(9200,) # print( y_train[:10]) #[5 3 1 5 4 1 4 4 2 3]
#print (x_train) 
#print (x_train.shape) #(9200, 178) #print (x_train.shape[1]) #178


y_train = np_utils.to_categorical(y_train, 2)
print(y_train.shape) #(9200, 6)

y_test = np_utils.to_categorical(y_test, 2)
print(y_test.shape)

#Feature Scaling
###x_train = np.reshape(x_train, (x_train.shape[0],1,X.shape[1]))
###x_test = np.reshape(x_test, (x_test.shape[0],1,X.shape[1]))

x_train = np.reshape(x_train, (x_train.shape[0],X.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],X.shape[1],1))

print(x_train)
print(x_train.shape)

model = Sequential()
model.add(LSTM(100, return_sequences = True, input_shape=(X.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100,return_sequences = True))
#model.add(Dropout(0.3))
#model.add(LSTM(100,return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dense(100, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))



model.compile(loss='binary_crossentropy',
              optimizer= "adam",
              metrics=["accuracy",mcor,recall, f1])
history = model.fit(x_train, y_train, epochs = 40, validation_data= (x_test, y_test))
model.evaluate(x_test, y_test)







####----------------------------------------- evaluate model-------------------------------------------------------------
#########    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
#history = model.fit(x_train, y_train, epochs=900, batch_size=40)

#history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=40)
#train = pd.DataFrame()
#val =pd.DataFrame()
#train[str(i)] = history.history['loss']
#val[str(i)] = history.history['val_loss']

## plot train and validation loss across multiple runs
#plt.plot(train, color='blue', label='train')
#plt.plot(val, color='orange', label='validation')
#plt.title('model train vs validation loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.show()
