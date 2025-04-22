#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing


# In[2]:


mult = 5
def load_file(path):
    data = pd.read_csv(path, sep=',')

    is_benign = data[' Label']=='BENIGN'
    flows_ok = data[is_benign]
    flows_ddos_full = data[~is_benign]

    sizeDownSample = len(flows_ok)*mult # tamanho do set final de dados anomalos

    # downsample majority
    if (len(flows_ok)*mult) < (len(flows_ddos_full)):
        flows_ddos_reduced = resample(flows_ddos_full,
                                         replace = False, # sample without replacement
                                         n_samples = sizeDownSample, # match minority n
                                         random_state = 27) # reproducible results
    else:
        flows_ddos_reduced = flows_ddos_full

    return flows_ok, flows_ddos_reduced


def load_huge_file(path):
    df_chunk = pd.read_csv(path, chunksize=500000)

    chunk_list_ok = []  # append each chunk df here
    chunk_list_ddos = []

    # Each chunk is in df format
    for chunk in df_chunk:
        # perform data filtering
        is_benign = chunk[' Label']=='BENIGN'
        flows_ok = chunk[is_benign]
        flows_ddos_full = chunk[~is_benign]

        if (len(flows_ok)*mult) < (len(flows_ddos_full)):
            sizeDownSample = len(flows_ok)*mult # tamanho do set final de dados anomalos

            # downsample majority
            flows_ddos_reduced = resample(flows_ddos_full,
                                             replace = False, # sample without replacement
                                             n_samples = sizeDownSample, # match minority n
                                             random_state = 27) # reproducible results
        else:
            flows_ddos_reduced = flows_ddos_full

        # Once the data filtering is done, append the chunk to list
        chunk_list_ok.append(flows_ok)
        chunk_list_ddos.append(flows_ddos_reduced)

    # concat the list into dataframe
    flows_ok = pd.concat(chunk_list_ok)
    flows_ddos = pd.concat(chunk_list_ddos)

    return flows_ok, flows_ddos


# In[3]:


# file 1
dataset_path = 'D:/amal/01-12/TFTP.csv'
flows_ok, flows_ddos = load_huge_file(dataset_path)
print('file 1 loaded')
# file 2
a,b = load_file('D:/amal/01-12/DrDoS_LDAP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 2 loaded')

# file 3
a,b = load_file('D:/amal/01-12/DrDoS_MSSQL.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 3 loaded')

# file 4
a,b = load_file('D:/amal/01-12/DrDoS_NetBIOS.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 4 loaded')

# file 5
a,b = load_file('D:/amal/01-12/DrDoS_NTP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 5 loaded')

# file 6
a,b = load_file('D:/amal/01-12/DrDoS_SNMP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 6 loaded')

# file 7
a,b = load_file('D:/amal/01-12/DrDoS_SSDP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 7 loaded')

# file 8
a,b = load_file('D:/amal/01-12/DrDoS_UDP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 8 loaded')

# file 9
a,b = load_file('D:/amal/01-12/Syn.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 9 loaded')

# file 10
a,b = load_file('D:/amal/01-12/DrDoS_DNS.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 10 loaded')

# file 11
a,b = load_file('D:/amal/01-12/UDPLag.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 11 loaded')

del a,b

samples = flows_ok.append(flows_ddos,ignore_index=True)
samples.to_csv('D:/amal/01-12/export_dataframe.csv', index = None, header=True)

del flows_ddos, flows_ok


# In[4]:


# file 1
flows_ok, flows_ddos = load_file('D:/amal/03-11/LDAP.csv')
print('file 1 loaded')

# file 2
a,b = load_file('D:/amal/03-11/MSSQL.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 2 loaded')

# file 3
a,b = load_file('D:/amal/03-11/NetBIOS.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 3 loaded')

# file 4
a,b = load_file('D:/amal/03-11/PortMap.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 4 loaded')

# file 5
a,b = load_file('D:/amal/03-11/Syn.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 5 loaded')
# file 6

a,b = load_file('D:/amal/03-11/UDP.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 6 loaded')

# file 7
a,b = load_file('D:/amal/03-11/UDPLag.csv')
flows_ok = flows_ok.append(a,ignore_index=True)
flows_ddos = flows_ddos.append(b,ignore_index=True)
print('file 7 loaded')

tests = flows_ok.append(flows_ddos,ignore_index=True)
tests.to_csv(r'D:/amal/01-12/export_tests.csv', index = None, header=True) 

del flows_ddos, flows_ok, a, b


# In[6]:


samples.head()


# In[11]:


samples.columns


# In[12]:


print('Dimensions of the  dataset:',samples.shape)


# In[7]:


samples.info()


# In[8]:


samples[' Label'].value_counts()
#class 1 not attack
#class 0 attack


# In[9]:


samples.sample(5)


# In[10]:


missing_values = samples.isnull().sum()
total_missing = missing_values.sum()
print("\nsamples_total_missing_in_data", total_missing)


# In[14]:


# training data
import numpy as np
# seed to remove randomness and reproduce results
np.random.seed(10)
samples = pd.read_csv('D:/Dataset/01-12/export_dataframe.csv', sep=',')
##########
def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)

# Flows Packet/s e Bytes/s - Replace infinity by 0
samples = samples.replace('Infinity','0')
samples = samples.replace(np.inf,0)
#samples = samples.replace('nan','0')
samples[' Flow Packets/s'] = pd.to_numeric(samples[' Flow Packets/s'])

samples['Flow Bytes/s'] = samples['Flow Bytes/s'].fillna(0)
samples['Flow Bytes/s'] = pd.to_numeric(samples['Flow Bytes/s'])


#Label
samples[' Label'] = samples[' Label'].replace('BENIGN',0)
samples[' Label'] = samples[' Label'].replace('DrDoS_DNS',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_LDAP',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_MSSQL',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_NTP',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_NetBIOS',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_SNMP',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_SSDP',1)
samples[' Label'] = samples[' Label'].replace('DrDoS_UDP',1)
samples[' Label'] = samples[' Label'].replace('Syn',1)
samples[' Label'] = samples[' Label'].replace('TFTP',1)
samples[' Label'] = samples[' Label'].replace('UDP-lag',1)
samples[' Label'] = samples[' Label'].replace('WebDDoS',1)

#Timestamp - Drop day, then convert hour, minute and seconds to hashing 
colunaTime = pd.DataFrame(samples[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
samples[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
del colunaTime,stringHoras


# flowID - IP origem - IP destino - Simillar HTTP -> Drop (individual flow analysis)
del samples[' Source IP']
del samples[' Destination IP']
del samples['Flow ID']
del samples['SimillarHTTP']
del samples['Unnamed: 0']

samples.to_csv(r'D:/Dataset/01-12/export_dataframe_proc.csv', index = None, header=True) 
print('Training data processed')


# In[15]:


####################### test data
tests = pd.read_csv('D:/Dataset/01-12/export_tests.csv', sep=',')
 
def string2numeric_hash(text):
    import hashlib
    return int(hashlib.md5(text).hexdigest()[:8], 16)

# Flows Packet/s e Bytes/s - Change infinity by 0
tests = tests.replace('Infinity','0')
tests = tests.replace(np.inf,0)
#amostras = amostras.replace('nan','0')
tests[' Flow Packets/s'] = pd.to_numeric(tests[' Flow Packets/s'])

tests['Flow Bytes/s'] = tests['Flow Bytes/s'].fillna(0)
tests['Flow Bytes/s'] = pd.to_numeric(tests['Flow Bytes/s'])


#Label
tests[' Label'] = tests[' Label'].replace('BENIGN',0)
tests[' Label'] = tests[' Label'].replace('LDAP',1)
tests[' Label'] = tests[' Label'].replace('NetBIOS',1)
tests[' Label'] = tests[' Label'].replace('MSSQL',1)
tests[' Label'] = tests[' Label'].replace('Portmap',1)
tests[' Label'] = tests[' Label'].replace('Syn',1)
#tests[' Label'] = tests[' Label'].replace('DrDoS_SNMP',1)
#tests[' Label'] = tests[' Label'].replace('DrDoS_SSDP',1)

#Timestamp - Drop day, then convert hour, minute and seconds to hashing 
colunaTime = pd.DataFrame(tests[' Timestamp'].str.split(' ',1).tolist(), columns = ['dia','horas'])
colunaTime = pd.DataFrame(colunaTime['horas'].str.split('.',1).tolist(),columns = ['horas','milisec'])
stringHoras = pd.DataFrame(colunaTime['horas'].str.encode('utf-8'))
tests[' Timestamp'] = pd.DataFrame(stringHoras['horas'].apply(string2numeric_hash))#colunaTime['horas']
del colunaTime,stringHoras

# flowID - IP origem - IP destino - Simillar HTTP -> Deletar (analise fluxo a fluxo)
del tests[' Source IP']
del tests[' Destination IP']
del tests['Flow ID']
del tests['SimillarHTTP']
del tests['Unnamed: 0']

tests.to_csv(r'D:/Dataset/01-12/export_tests_proc.csv', index = None, header=True) 
print('Test data processed')


# In[16]:


def train_test(samples):
    # Import `train_test_split` from `sklearn.model_selection`
    #CROSS validation
    from sklearn.model_selection import train_test_split
    X=samples.iloc[:,0:(samples.shape[1]-1)]

    y= samples.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Split the data into training, validation, and test datasets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
 
# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Make predictions on the validation dataset
y_pred = model.predict(X_val)
 
# Evaluate the model performance on the validation dataset
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")
 
# Make predictions on the test dataset
y_pred_test = model.predict(X_test)
 
# Evaluate the model performance on the test dataset
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy_test}")


# In[17]:


def normalize_data(X_train,X_test):
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    
    # Define the scaler 
    #scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    
    # Scale the train set
    X_train = scaler.transform(X_train)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    return X_train, X_test


# In[18]:


# Reshape data input

def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def format_2d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1]))


# In[19]:


from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from keras.layers import LeakyReLU
from keras.optimizers import Adam


# In[20]:


#input_size
# -> CIC-DDoS2019 82
# -> CIC-IDS2018 78

def GRU_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(GRU(32, input_shape=(input_size,1), return_sequences=False)) #
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.build()
    print(model.summary())
    
    return model


# In[21]:


def CNN_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(input_size,1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=32, kernel_size=16, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model


# In[22]:


def DNN_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(Dense(2, activation='relu', input_shape=(input_size,)))
    #model.add(Dense(100, activation='relu'))   
    #model.add(Dense(40, activation='relu'))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model


# In[23]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

def BiLSTM_model(input_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(input_size, 1)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())

    return model


# In[ ]:





# In[24]:


def ANN_model(input_size):
    model = Sequential()
    model.add(Dense(64, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    return model

# Example usage
#input_size = 10
#DNN_model = DNN_model(input_size)


# In[25]:


def LSTM_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(LSTM(32,input_shape=(input_size,1), return_sequences=False))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    return model


# In[26]:


from tensorflow import keras
from tensorflow.keras import layers

def G_C_hybrid_model():
    model = keras.Sequential()
    
    # Conv1D layers
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_size,1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(Dropout( 0.4556))
    # LSTM layer
   # model.add(layers.LSTM(64, return_sequences=True))
    
    # GRU layer
    model.add(layers.GRU(28, return_sequences=True))
    model.add(Dropout(0.5))
    # Flatten layer
    #model.add(layers.Flatten())
    
    # Dense layers
    
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    
    return model


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, GRU, Dropout 

def CNN_GRU_model(input_size):

    model = Sequential()
    
    # CNN Layers
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', input_shape=(input_size, 1)))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Conv1D(filters=32, kernel_size=16, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten output for GRU
    model.add(Flatten())
    
    # GRU Layer
    model.add(GRU(units=32, activation='relu'))
    
    # Dense Layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    return model


# In[28]:


def GRU_BiLSTM_model(input_size):
    model = Sequential()
    
    # GRU Layer
    model.add(GRU(64, return_sequences=True, input_shape=(input_size,1)))
    
    # Bidirectional LSTM Layer
    model.add(Bidirectional(LSTM(32)))
    
    # Fully Connected Layers
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    
    return model


# In[29]:


def SVM():
    return SVC(kernel='linear')


# In[30]:


from sklearn.ensemble import GradientBoostingClassifier
def gradient_boosting():
    return GradientBoostingClassifier(n_estimators=100, random_state=42)


# In[31]:


from sklearn.ensemble import RandomForestClassifier
def random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


# In[32]:


from sklearn.tree import DecisionTreeClassifier
def decision_tree():
    return DecisionTreeClassifier(random_state=42)


# In[33]:


import xgboost
#xgb_clf = xgboost.XGBRFClassifier(max_depth=3, random_state=1)
from xgboost import XGBClassifier 
def xgboost():
    return XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)


# In[34]:


def LR():
    return LogisticRegression()


# In[35]:


def GD():
    return SGDClassifier()


# In[36]:


def kNN():
    return KNeighborsClassifier(n_neighbors=3, n_jobs=-1)


# In[37]:


from sklearn.naive_bayes import GaussianNB
def gnb():
    return GaussianNB()


# In[38]:


from sklearn.ensemble import ExtraTreesClassifier
def Model_ETC():
    return ExtraTreesClassifier(n_estimators=10, random_state=42)


# In[39]:


# compile and train learning model
import matplotlib.pyplot as plt
def compile_train(model, X_train, y_train, X_val, y_val, deep=True):
    
    start_time = time.time()
    
    if(deep==True):
        model.compile(
            loss='binary_crossentropy', 
            optimizer='adam',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train, 
            y_train,
            epochs=10, 
            batch_size=256, 
            verbose=1,
            validation_data=(X_val, y_val)
        )
             
# Plot the accuracy graph
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        epochs = range(1, len(accuracy) + 1)

        plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
# Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        print(model.metrics_names)
    else:
         model.fit(X_train, y_train)

    end_time = time.time()
    train_time = end_time - start_time
    
    start_time = time.time()
    loss, accuracy = model.evaluate(X_val, y_val)
    end_time = time.time()
    test_time = end_time - start_time

    print(f'Training Time: {train_time:.3f}s') 
    print(f'Testing Time: {test_time:.3f}s')
         
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
    
    return model
import matplotlib.pyplot as plt


# In[40]:


def testes(model, X_test, y_test, y_pred, deep=True):
    
    start_time = time.time()
    
    if (deep == True):
        score = model.evaluate(X_test, y_test, verbose=1)
        print(score)

    # Alguns testes adicionais
    # y_test = formatar2d(y_test)
    # y_pred = formatar2d(y_pred)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy')
    print(acc)

    # Precision
    prec = precision_score(y_test, y_pred)  # ,average='macro')
    print('\nPrecision')
    print(prec)

    # Recall
    rec = recall_score(y_test, y_pred)  # ,average='macro')
    print('\nRecall')
    print(rec)

    # F1 score
    f1 = f1_score(y_test, y_pred)  # ,average='macro')
    print('\nF1 Score')
    print(f1)

    # False Positive Rate (FPR)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    print('\nFalse Positive Rate (FPR)')
    print(fpr)

    # Average
    avrg = (acc + prec + rec + f1 + fpr) / 5
    print('\nAverage (acc, prec, rec, f1, FPR)')
    print(avrg)

    end_time = time.time()
    train_time = end_time - start_time
    test_time = end_time - start_time

    print(f'\nTesting Time: {test_time:.3f}s')
    print(f'Training Time: {train_time:.3f}s') 
    return acc, prec, rec, f1, fpr, avrg, test_time, train_time


# In[41]:


def test_normal_atk(y_test,y_pred):
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    
    normal = len(df.query('y_test == 0'))
    atk = len(y_test)-normal
    
    wrong = df.query('y_test != y_pred')
    
    normal_detect_rate = (normal - wrong.groupby('y_test').count().iloc[0][0]) / normal
    atk_detect_rate = (atk - wrong.groupby('y_test').count().iloc[1][0]) / atk
    
    #print(normal_detect_rate,atk_detect_rate)
    
    return normal_detect_rate, atk_detect_rate


# In[42]:


# Save model and weights

def save_model(model,name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    model_json = model.to_json()
    with open(arq_json,"w") as json_file:
        json_file.write(model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    model.save_weights(arq_h5)
    print('Model Saved')
    
def load_model(name):
    from keras.models import model_from_json
    
    arq_json = 'Models/' + name + '.json'
    json_file = open(arq_json,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    arq_h5 = 'Models/' + name + '.h5'
    loaded_model.load_weights(arq_h5)
    
    print('Model loaded')
    
    return loaded_model

def save_Sklearn(model,nome):
    import pickle
    arquivo = 'Models/'+ nome + '.pkl'
    with open(arquivo,'wb') as file:
        pickle.dump(model,file)
    print('Model sklearn saved')

def load_Sklearn(nome):
    import pickle
    arquivo = 'Models/'+ nome + '.pkl'
    with open(arquivo,'rb') as file:
        model = pickle.load(file)
    print('Model sklearn loaded')
    return model


# In[43]:


# UPSAMPLE OF NORMAL FLOWS
    
samples = pd.read_csv('D:/Dataset/01-12/export_dataframe_proc.csv', sep=',')

X_train, X_test, y_train, y_test = train_test(samples)
#5 croos 20 param

#junta novamente pra aumentar o numero de normais
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
is_benign = X[' Label']==0 #base de dados toda junta

normal = X[is_benign]
ddos = X[~is_benign]

# upsample minority
normal_upsampled = resample(normal,
                          replace=True, # sample with replacement
                          n_samples=len(ddos), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([normal_upsampled, ddos])

# Specify the data 
X_train=upsampled.iloc[:,0:(upsampled.shape[1]-1)]    #DDoS
y_train= upsampled.iloc[:,-1]  #DDoS

input_size = (X_train.shape[1], 1)

del X, normal_upsampled, ddos, upsampled, normal #, l1, l2


# In[45]:


tests = pd.read_csv('D:/amal/01-12/export_tests_proc.csv', sep=',')

# X_test = np.concatenate((X_test,(tests.iloc[:,0:(tests.shape[1]-1)]).to_numpy())) # testar 33% + dia de testes
# y_test = np.concatenate((y_test,tests.iloc[:,-1]))

del X_test,y_test                            # testar só o dia de testes
X_test = tests.iloc[:,0:(tests.shape[1]-1)]                        
y_test = tests.iloc[:,-1]

# print((y_test.shape))
# print((X_test.shape))

X_train, X_test = normalize_data(X_train,X_test)


# In[46]:


results = pd.DataFrame(columns=['Method','Accuracy','Precision','Recall', 'F1_Score', 'Average',
                                'Normal_Detect_Rate','Atk_Detect_Rate',
                                'Training_Time', 'Testing_Time'])


# In[47]:


from keras.layers import LSTM
model_Bilstm = BiLSTM_model(82)


# In[48]:


import time
model_Bilstm = compile_train(model_Bilstm,format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[49]:


y_pred = model_Bilstm.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_Bilstm,format_3d(X_test),y_test,y_pred)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'model_Bilstm', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def BiLSTM_model(input_size):
    model = Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(input_size, 1)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
    
    # Generate the model plot
    plot_model(BiLSTM_model, to_file='BiLSTM_model_architecture.png', show_shapes=True, show_layer_names=True)
    
    # Display the plot
    img = plt.imread('BiLSTM_model_architecture.png')
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    return model

# Example usage
input_size = 82  # Replace with your actual input size
model = BiLSTM_model(input_size)


# In[61]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.bar(['FPR', 'Accuracy'], [fpr * 100, acc *100], color=['blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('False Positive Rate (FPR) and Accuracy')
plt.ylim([0, 1])
plt.show()


# In[53]:


gru_bilstm = GRU_BiLSTM_model(82)


# In[55]:


gru_bilstm = compile_train(gru_bilstm, format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[56]:


y_pred = gru_bilstm.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(gru_bilstm,format_3d(X_test),y_test,y_pred)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'GRU_Bilstm', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[ ]:





# In[68]:


from tensorflow.keras.layers import Dropout
input_size = 82
model_CNN_GRU = G_C_hybrid_model()


# In[72]:


model_CNN_GRU = compile_train(model_CNN_GRU,format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[175]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.bar(['FPR', 'Accuracy'], [fpr * 100, acc *100], color=['blue', 'green'])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('False Positive Rate (FPR) and Accuracy')
plt.ylim([0, 1])
plt.show()


# In[73]:


from keras.models import Sequential 
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model_cnn = CNN_model(82)


# In[74]:


model_cnn = compile_train(model_cnn,format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[75]:


y_pred = model_cnn.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_cnn,format_3d(X_test),y_test,y_pred)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'model_cnn', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[79]:


def CNN_GRU_model(input_size):

    model = Sequential()
    
    model.add(Conv1D(64, 8, activation='relu', input_shape=(input_size,1), 
                     return_sequences=True))
    
    model.add(Conv1D(32, 16, activation='relu',  
                     return_sequences=True))
    
    # Rest of model
    
    return model
model_H_CNN_GRU = CNN_GRU_model(32)


# In[80]:


model_H_CNN_GRU = compile_train(model_H_CNN_GRU,format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[ ]:


y_pred = model_H_CNN_GRU.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_cnn,format_3d(X_test),y_test,y_pred)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'model_H_CNN_GRU', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[81]:


import keras
from keras import layers

def devcnn_model(input_shape):
    model = keras.Sequential()
    
    model.add(layers.Conv1D(32, kernel_size=16, activation='relu', input_shape=input_shape))  
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())  
    return model
 


# In[82]:


def build_hybrid_model():
    model = keras.Sequential()
    hybrid_model = build_hybrid_model(82)
    # Conv1D layers
    model.add(layers.Conv1D(32, kernel_size=16, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # LSTM layer
    model.add(layers.LSTM(64, return_sequences=True))
    
    # GRU layer
    model.add(layers.GRU(32, return_sequences=True))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


# In[86]:


hybrid_model = build_hybrid_model(82)


# In[87]:


CNN_DEV = devcnn_model(input_shape=(82, 1))


# In[88]:


CNN_DEV = compile_train(CNN_DEV, format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[90]:


y_pred = CNN_DEV.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(CNN_DEV,format_3d(X_test),y_test,y_pred)
#acc,fpr, prec, rec, f1, avrg = testes(model_cnn, format_3d(X_train), y_train, format_3d(X_test), y_test)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'CNN_DEV', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk,'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[91]:


from keras.layers import GRU
model_gru = GRU_model(82)


# In[92]:


model_gru = compile_train(model_gru, format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[93]:


y_pred = model_gru.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_gru,format_3d(X_test),y_test,y_pred)
#acc,fpr, prec, rec, f1, avrg = testes(model_cnn, format_3d(X_train), y_train, format_3d(X_test), y_test)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'GRU', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[94]:


model_dnn = DNN_model(X_train.shape[1])


# In[95]:


model_dnn = compile_train(model_dnn,format_3d(X_train), y_train, format_3d(X_test), y_test)


# In[96]:


y_pred = model_dnn.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_dnn,format_3d(X_test),y_test,y_pred)
#acc,fpr, prec, rec, f1, avrg = testes(model_cnn, format_3d(X_train), y_train, format_3d(X_test), y_test)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'DNN', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[97]:


model_ann = ANN_model(71)


# In[99]:


from tensorflow import keras
from tensorflow.keras import layers

def build_hybrid_model():
    model = keras.Sequential()
    
    # Conv1D layers
    model.add(layers.Conv1D(32, kernel_size=16, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # LSTM layer
    model.add(layers.LSTM(64, return_sequences=True))
    
    # GRU layer
    model.add(layers.GRU(32, return_sequences=True))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Build the hybrid model
model = build_hybrid_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}') 
print(f'Test Accuracy: {accuracy:.4f}')


# In[100]:


# Plot the accuracy graph
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[101]:





# In[102]:


y_pred = model.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model,format_3d(X_test),y_test,y_pred)
#acc,fpr, prec, rec, f1, avrg = testes(model_cnn, format_3d(X_train), y_train, format_3d(X_test), y_test)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'CNN_LSTM_GRU', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[11]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def devcnn_model(input_shape):
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=16, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(16, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define input shape (you need to specify this based on your data)
input_shape = (82, 1)  # Assuming 82 features and 1 channel

# Build the model
model = devcnn_model(input_shape)

# Plot the model
plot_model(model, to_file='model-CNN_visualization.png', show_shapes=True, show_layer_names=True)

# Display the generated image using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(plt.imread('model-CNN_visualization.png'))
plt.axis('off')
plt.show()


# In[1]:


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def build_hybrid_model():
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=16, activation='relu', input_shape=(82, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(16, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(64, return_sequences=True),
        layers.GRU(32, return_sequences=True),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Build the model
model = build_hybrid_model()

# Plot the model
plot_model(model, to_file='model_visualizationCGL.png', show_shapes=True, show_layer_names=True)

# Display the generated image using matplotlib
plt.figure(figsize=(15, 12))
plt.imshow(plt.imread('model_visualizationCGL.png'))
plt.axis('off')
plt.show()


# In[110]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
input_size = 82
def BiLSTM_model(input_size):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(input_size, 1)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
 
    return model

# Build the hybrid model
model = BiLSTM_model(82)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}') 
print(f'Test Accuracy: {accuracy:.4f}')
# Plot the accuracy graph
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[184]:


y_pred = BiLSTM_model.predict(format_3d(X_test)) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(BiLSTM_model,format_3d(X_test),y_test,y_pred)
#acc,fpr, prec, rec, f1, avrg = testes(model_cnn, format_3d(X_train), y_train, format_3d(X_test), y_test)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':' BiLSTM_model2', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense
from keras.utils import plot_model
from keras.models import Sequential
input_size = 82
def BiLSTM_model(input_size):
    model = keras.Sequential()
    model.add(Bidirectional(LSTM(32), input_shape=(input_size, 1)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    print(model.summary())
 
    return model

# Build the hybrid model
model = BiLSTM_model(82)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}') 
print(f'Test Accuracy: {accuracy:.4f}')
# Plot the accuracy graph
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[132]:


y_pred = model_gnb.predict(X_test) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_gnb,X_test,y_test,y_pred,False)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'NB', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[133]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
samples = y_test  # Actual labels
y_pred = model_gnb.predict(X_test) 

# Use y_test here instead of y_true
cm = confusion_matrix(y_test, y_pred)  

plt.figure(figsize=(6, 6))
ax = plt.subplot()
sns.heatmap(cm, 
            annot=True,
            fmt='g',
            ax=ax,
            cmap=plt.cm.viridis);

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix For naive_bayes'); 
ax.xaxis.set_ticklabels(['Normal', 'Attack']) 
ax.yaxis.set_ticklabels(['Normal', 'Attack']);


# In[135]:


import matplotlib.pyplot as plt

y_pred = model_gnb.predict(X_test) 
y_pred = y_pred.round()

normal_count = (y_pred == 0).sum()
attack_count = (y_pred == 1).sum()

plt.bar([0, 1], [normal_count, attack_count])
plt.xticks([0, 1], ['Normal', 'Attack'])
plt.ylabel('Count')
plt.title('Predicted Normal vs Attack')

plt.show()


# In[136]:


from sklearn.linear_model import LogisticRegression
model_lr = LR()


# In[137]:


model_lr.fit(X_train, y_train) 
y_pred = model_lr.predict(X_test)


# In[ ]:





# In[138]:


y_pred = model_lr.predict(X_test) 

y_pred = y_pred.round()
 
acc,fpr, prec, rec, f1, avrg, test_time, train_time = testes(model_lr,X_test,y_test,y_pred,False)

norm, atk = test_normal_atk(y_test,y_pred)

results = results.append({'Method':'LR', 'Accuracy':acc, 'Precision':prec, 'F1_Score':f1,'False Positive Rate (FPR)':fpr,
                          'Recall':rec,'Average':avrg, 'Normal_Detect_Rate':norm, 'Atk_Detect_Rate':atk, 'Test_Time':test_time , 'Train_Time':train_time}, ignore_index=True)


# In[139]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
samples = y_test  # القيم الحقيقية Labels
y_pred = model_lr.predict(X_test) # القيم المتنبأ بها Predicted

 
cm = confusion_matrix(y_test, y_pred) 

plt.figure(figsize=(6, 6))
ax = plt.subplot()
sns.heatmap(cm, 
            annot=True,  
            fmt='g',
            ax=ax,
            cmap=plt.cm.inferno); #defines heatmap

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix LogisticRegression'); 
ax.xaxis.set_ticklabels(['Normal', 'Attack']) 
ax.yaxis.set_ticklabels(['Normal', 'Attack']);


# In[164]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[165]:


print(y_train.value_counts())#368,604


# In[166]:


print(y_test.value_counts())#298,578


# In[183]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models, names, X_test, y_test):
    plt.figure(figsize=(10, 8))

    for model, name in zip(models, names):
        
        y_pred_prob = model.predict(X_test)

        #  
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

         
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')

     
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

models = [  model_cnn, CNN_DEV, model_gru, model_dnn, build_hybrid_model]  # قائمة النماذج
names = [ "CNN", "CNN_DEV", "GRU", "DNN", "CNN_LSTM_GRU"]  # أسماء النماذج
plot_roc_curves(models, names, X_test, y_test)


# In[171]:


ax = sns.catplot(data=results.iloc[:,:5].query('Method != "LSTM" and Method != "CNN"'), col='Method', col_wrap=3, kind='bar', height=3, aspect=2)
ax.set(ylim=(0.99,1))
ax.set_xticklabels(rotation=45)
ax = ax


# In[186]:


# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[188]:


from sklearn.feature_selection import chi2 
from sklearn.feature_selection import SelectKBest 
from sklearn.ensemble import ExtraTreesClassifier

selecting 20 best features
elect_best= SelectKBest(chi2, k=20)
X_feat_20 = select_best.fit_transform(data_X, data_y_trans)
X_feat_20.shape


# In[189]:


model.feature_importances_


# In[190]:


# DRAWN feature
feature_importance_std = pd.Series(model.feature_importances_, index=data_X.columns)
feature_importance_std.nlargest(20).plot(kind='bar', title='Standardised Dataset Feature Selection using ExtraTreesClassifier')


# In[191]:


# RoC curve Function 

def RoC_Curve(classifier, X_val, y_val, title): 
        """ RoC Curve for Classifier 
        Parameters: 
        ------------
        classifier: Machine Learning Classifier to be Evaluated
        X_val: Validation Dataset
        y_val: Label/Target of Validation Dataset

        Attributes:
        Plots the Graph    
        
        Note: Some part of this Method code is taken 
            from Sklearn Website
        """

        lw = 2
        n_classes = 12
        y_test1 = to_categorical(y_val)
        pred_RFC_proba = classifier.predict_proba(X_val)
        y_score = pred_RFC_proba

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(20,10))
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        list_class = ['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'UDP-lag', 'WebDDoS']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(list_class[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title) 
        plt.legend(loc="lower right")
        plt.show()



# In[192]:


from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 


# In[193]:


title = 'Receiver operating characteristic of Random Forest'
RoC_Curve(rf, X_test_std_20, y_test_20, title)


# In[194]:


print("Classification Report for Decision Tree: \n", classification_report(le.inverse_transform(y_test_20), le.inverse_transform(dt_y_pred)))


# In[195]:


dt_conf_mat = confusion_matrix(y_test_20, dt_y_pred)
print("Decision Tree Confusion: \n", dt_conf_mat)


# In[196]:


acc_score_dt = accuracy_score(y_test_20, dt_y_pred)
print("Accuracy Score for Decision Tree: \n", acc_score_dt*100)


# In[ ]:


# RoC Curve 
title = 'Receiver operating characteristic of Decision Tree'
RoC_Curve(dt, X_test_std_20, y_test_20, title)


# In[ ]:


from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn

def compile_train(model, X_train, y_train, X_val, y_val, deep=True):

    # Training code....
    
    # Evaluate model
    y_pred = model.predict(X_val)
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred > 0.5)  
    df_cm = pd.DataFrame(cm, index=[i for i in "YN"],
                  columns=[i for i in "YN"])
    plt.figure(figsize=(5,5))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    
    return model


# In[ ]:




