import pandas as pd #for reading dataset
import numpy as np # array handling functions
from time import sleep

dataset = pd.read_csv("Book1.csv")#reading dataset
x = dataset.iloc[:,:-1].values #locating inputs
y = dataset.iloc[:,-1].values #locating outputs

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#printing X and Y
print("x=",x)
print("y=",y)

from sklearn.model_selection import train_test_split # for splitting dataset
x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25 ,random_state = 0)
#printing the spliited dataset
print("x_train=",x_train)
print("x_test=",x_test)
print("y_train=",y_train)
print("y_test=",y_test)
 #importing algorithm
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(x_train,y_train)#trainig Algorithm

y_pred=classifier.predict(x_test) #testing model
print("y_pred",y_pred) # predicted output
try:
    import serial
    ser = serial.Serial('COM4',baudrate=9600,timeout=0.3)
    ser.flushInput()
    A=1
    B=1
    while True:
        a=ser.readline().decode('ascii') # reading serial data
        print(a)
        b=a
        for letter in b:
            if(letter =='W'):
                D1 =b[1]+b[2]+b[3]+b[4]
                print("wind : ",D1)
                a1 =int(D1)

            if(letter =='T'):
                D2 =b[6]+b[7]+b[8]
                print("temperature : ",D2)
                a2 =int(D2)
            if(letter =='H'):
                D3 =b[10]+b[11]+b[12]
                print(" humility : ",D3)
                a3 =int(D3)
            


                ##PREDICTED OUTPUT
                OUTPUT = classifier.predict([[a1,a2,a3]])
                print('DECISION TREE OUTPUT: ',OUTPUT)
                                
              
                
                if OUTPUT ==1:                  
                    print("Rainy")

                elif OUTPUT==2:
                    print("Winter")
                elif OUTPUT==3:
                    print("Summer")
                





except Exception as e:
    print(e)
