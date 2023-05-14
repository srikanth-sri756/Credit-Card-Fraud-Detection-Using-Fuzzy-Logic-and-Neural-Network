from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM  #class for LSTM regression
from keras.layers import Dropout
from keras.models import model_from_json
from  sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn import linear_model
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf

main = tkinter.Tk()
main.title("Credit Card Fraud Detection Using Fuzzy Logic and Neural Network")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global model
global filename, dataset
global X, Y
mse = []

classifier = linear_model.LogisticRegression(max_iter=1000)

def getFrequency(frequency, card, date_time):
    count = 0
    for i in range(len(frequency)):
        if str(frequency[i,0]) == str(date_time) and str(frequency[i,1]) == card:
            count = frequency[i,2]
            break
    return count

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")
    dataset = pd.read_csv(filename,nrows=100000)
    dataset['trans_date_trans_time'] = pd.to_datetime(dataset['trans_date_trans_time'])
    text.insert(END,str(dataset.head()))
    

def runFuzzyMemberFunction():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("Data.csv"):
        dataset = pd.read_csv("Data.csv")
    else:
        X = []
        card = np.unique(dataset['cc_num'])
        for i in range(len(card)):
            data = dataset.loc[dataset['cc_num'] == card[i]]
            frequency = data.groupby(['trans_date_trans_time', 'cc_num']).size()
            frequency = frequency.to_frame(name = 'frequency').reset_index()
            frequency = frequency.values
            average_time = data['trans_date_trans_time'].mean().to_datetime64().astype('M8[D]').astype('O')
            average_amount = data['amt'].mean()
            amounts = data['amt'].ravel()
            locations = data['state'].ravel()
            data = data.values
            last_transaction_date = data[len(data)-1,1].to_datetime64().astype('M8[D]').astype('O')
            for j in range(len(data)):
                days_difference = average_time - data[j,1].to_datetime64().astype('M8[D]').astype('O') #average transaction time - current time
                days_difference = days_difference.days
                amount_difference = average_amount - amounts[j] #average transaction amount - current transaction amount
                location = locations[j] #finding location  inside or outside toronto and canada
                interval = last_transaction_date  - data[j,1].to_datetime64().astype('M8[D]').astype('O') #differnce between last transaction date and current transaction date
                interval = interval.days
                freq = getFrequency(frequency, card[i], data[j,1])
                temp = []
                temp.append(card[i])
                if days_difference < 4:
                    temp.append(0) #adding 0 means LOW
                if days_difference >= 4 and days_difference < 7:
                    temp.append(1)#adding 1 means MEDIUM
                if days_difference >= 7:
                    temp.append(2) #adding 2 means HIGH
                if amount_difference < 10:
                    temp.append(0)
                if amount_difference >= 10 and amount_difference < 50:
                    temp.append(1)
                if amount_difference >= 50:
                    temp.append(2)
                if location == 'TN':
                    temp.append(0)
                if location == 'CA':
                    temp.append(1)
                if location != 'TN' and location != 'CA':
                    temp.append(2)
                if interval < 30:
                    temp.append(0)
                if interval >= 30 and interval < 90:
                    temp.append(1)
                if interval >= 90:
                    temp.append(2)
                if freq < 3:
                    temp.append(0)
                if freq >= 3 and freq < 5:
                    temp.append(1)
                if freq >= 5:
                    temp.append(2)
                X.append(temp)    
        output = pd.DataFrame(X, columns=['Card_No', 'Time_Difference', 'Amount_Difference', 'Location', 'Interval', 'Frequency'])
        temp = output.drop(['Card_No'], axis = 1)
        temp = temp.values
        labels = []
        for i in range(len(temp)):
            unique, count = np.unique(temp[i], return_counts=True)
            index = np.argmax(count)
            if index == 0:
                labels.append("Legal")
            if index == 1:
                labels.append("Suspicious")
            if index == 2:
                labels.append("Fraud")
        
        output['label'] = labels
        output.to_csv("Data.csv", index= False)
        dataset = pd.read_csv("Data.csv")
        fraud = dataset.loc[dataset['label'] == "Fraud"]
        suspicious = dataset.loc[dataset['label'] == "Suspicious"]
        legal = dataset.loc[dataset['label'] == "Legal"]
        fraud.to_csv("fraud.csv", index=False)
        suspicious.to_csv("suspicious.csv", index=False)
        legal.to_csv("legal.csv", index=False)
        fraud = pd.read_csv("fraud.csv", nrows=9000)
        suspicious = pd.read_csv("suspicious.csv", nrows=10000)
        legal = pd.read_csv("legal.csv", nrows=10000)
        dataset = [fraud, suspicious, legal]
        dataset = pd.concat(dataset)
        dataset.to_csv("Data.csv", index=False)            
        os.remove("fraud.csv")       
        os.remove("suspicious.csv")       
        os.remove("legal.csv")
        dataset = pd.read_csv("Data.csv")
    text.insert(END,str(dataset.head())+"\n\n")
    label = dataset.groupby('label').size()
    le = LabelEncoder()
    dataset['label'] = pd.Series(le.fit_transform(dataset['label'].astype(str)))
    dataset.drop(['Card_No'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]        
    sc = MinMaxScaler(feature_range = (0, 1))
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    label.plot(kind="bar")
    plt.title("Different Transactions Graph")
    plt.show()


def runFuzzy():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test, mse
    text.delete('1.0', END)
    mse.clear()
    fuzzy = FuzzyKMeans(k=3, m=2)
    fuzzy.fit(X)
    predict = fuzzy.labels_
    print(predict)
    fuzzy_mse = 1.0 - accuracy_score(Y,predict)    
    mse.append(fuzzy_mse)
    text.insert(END,"Fuzzy Logic MSE : "+str(fuzzy_mse)+"\n\n")

def runLSTM():
    global X, Y, dataset
    global X_train, X_test, y_train, y_test, mse
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_train1.shape)
    if os.path.exists("model/lstm_model.json"):
        with open('model/lstm_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            lstm = model_from_json(loaded_model_json)
        json_file.close()
        lstm.load_weights("model/lstm_weights.h5")
        lstm._make_predict_function()   
    else:
        lstm = Sequential()#defining sqeuential object
        #adding LSTM layer to sequential deep learning object with 50 as hidden layers to filter dataset givem as Xtrain input
        lstm.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
        #defining dropout layer to remove irrelevant values from dataset
        lstm.add(Dropout(0.2))
        #defiing another layer
        lstm.add(LSTM(units = 50, return_sequences = True))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(units = 50, return_sequences = True))
        lstm.add(Dropout(0.2))
        lstm.add(LSTM(units = 50))
        lstm.add(Dropout(0.2))
        #defining output layer as Y labels
        lstm.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
        #compile the model
        lstm.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #start training model
        hist = lstm.fit(X_train1, y_train1, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test1, y_test1))
        lstm.save_weights('model/lstm_weights.h5')#save the model            
        model_json = lstm.to_json()
        with open("model/lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(lstm.summary())
    predict = lstm.predict(X_test1) #perform predition on test data
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    lstm_mse = 1.0 - accuracy_score(y_test1,predict) #calculate MSE value
    mse.append(lstm_mse)
    text.insert(END,"LSTM MSE : "+str(lstm_mse)+"\n\n")

def f_per_particle(m, alpha):
    global X_train
    global y_train
    global classifier
    total_features = 5
    if np.count_nonzero(m) == 0:
        X_subset = X_train
    else:
        X_subset = X_train[:,m==1]
    classifier.fit(X_subset, y_train)
    P = (classifier.predict(X_subset) == y_train).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def runPSOLSTM():
    global X, Y, dataset, f
    global X_train, X_test, y_train, y_test, mse
    print(X_train.shape)
    text.insert(END,"Features Found in dataset before applying PSO = "+str(X_train.shape[1])+"\n")
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
    dimensions = 5 # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pos = optimizer.optimize(f, iters=2)#OPTIMIZING FEATURES
    pos = np.load("model/pos.npy")
    X_train1 = X_train[:,pos==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
    X_test1 = X_test[:,pos==1]
    text.insert(END,"Features Found in dataset after applying PSO = "+str(X_train1.shape[1])+"\n")
    print(X_train1.shape)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))
    X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
    print(X_train1.shape)

    if os.path.exists("model/pso_lstm_model.json"):
        with open('model/pso_lstm_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            pso_lstm = model_from_json(loaded_model_json)
        json_file.close()
        pso_lstm.load_weights("model/pso_lstm_weights.h5")
        pso_lstm._make_predict_function()   
    else:
        pso_lstm = Sequential()#defining sqeuential object
        #adding LSTM layer to sequential deep learning object with 50 as hidden layers to filter dataset givem as Xtrain input
        pso_lstm.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train1.shape[1], X_train1.shape[2])))
        #defining dropout layer to remove irrelevant values from dataset
        pso_lstm.add(Dropout(0.2))
        #defiing another layer
        pso_lstm.add(LSTM(units = 50, return_sequences = True))
        pso_lstm.add(Dropout(0.2))
        pso_lstm.add(LSTM(units = 50, return_sequences = True))
        pso_lstm.add(Dropout(0.2))
        pso_lstm.add(LSTM(units = 50))
        pso_lstm.add(Dropout(0.2))
        #defining output layer as Y labels
        pso_lstm.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
        #compile the model
        pso_lstm.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #start training model
        hist = pso_lstm.fit(X_train1, y_train1, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test1, y_test1))
        pso_lstm.save_weights('model/pso_lstm_weights.h5')#save the model            
        model_json = pso_lstm.to_json()
        with open("model/pso_lstm_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/pso_lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = pso_lstm.predict(X_test1) #perform predition on test data
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    lstm_mse = 1.0 - accuracy_score(y_test1,predict) #calculate MSE value
    mse.append(lstm_mse)
    text.insert(END,"PSO LSTM MSE : "+str(lstm_mse)+"\n\n")

def trainingGraph():
    f = open('model/lstm_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    train_mse = data['loss']
    test_mse = data['val_loss']

    f = open('model/pso_lstm_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    pso_mse = data['loss']
    pso_test_mse = data['val_loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Square Error')
    plt.plot(train_mse, 'ro-', color = 'green')
    plt.plot(test_mse, 'ro-', color = 'blue')
    plt.plot(pso_mse, 'ro-', color = 'yellow')
    plt.plot(pso_test_mse, 'ro-', color = 'orange')
    plt.legend(['Train MSE', 'Test MSE','PSO Train MSE', 'PSO TEST MSE'], loc='upper left')
    plt.title('LSTM & PSO MSE Performance Graph')
    plt.show()

def mseGraph():
    global mse
    height = mse
    bars = ('Fuzzy Logic MSE', 'LSTM MSE', 'PSO MSE')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Fuzzy Logic VS LSTM MSE Comparison Graph")
    plt.show()

def close():
    main.destroy()

font = ('times', 14, 'bold')
title = Label(main, text='Credit Card Fraud Detection Using Fuzzy Logic and Neural Network')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Credit Card Fraud Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

memberButton = Button(main, text="Calculate Fuzzy Membership Functions", command=runFuzzyMemberFunction)
memberButton.place(x=50,y=150)
memberButton.config(font=font1)

fuzzyButton = Button(main, text="Run Fuzzy Logic Algorithm", command=runFuzzy)
fuzzyButton.place(x=50,y=200)
fuzzyButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=50,y=250)
lstmButton.config(font=font1)

lstmButton = Button(main, text="Run PSO-LSTM Algorithm", command=runPSOLSTM)
lstmButton.place(x=50,y=300)
lstmButton.config(font=font1)

lstmgraphButton = Button(main, text="LSTM Training Graph", command=trainingGraph)
lstmgraphButton.place(x=50,y=350)
lstmgraphButton.config(font=font1)

msegraphButton = Button(main, text="MSE Comparison Graph", command=mseGraph)
msegraphButton.place(x=50,y=400)
msegraphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
