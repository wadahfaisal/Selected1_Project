#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

#Data_preprocessing

#load  the dataset
dataset=pd.read_csv(r'C:\Users\Salma\Downloads\heart.csv')

#split dataset into x,y
x=pd.DataFrame(dataset.iloc[:,0:11].values)
y=pd.DataFrame(dataset.iloc[:,11].values)

#Encode categorical data
labelencoder_x_6=LabelEncoder()
x.loc[:,6]=labelencoder_x_6.fit_transform(x.iloc[:,6])
labelencoder_x_1=LabelEncoder()
x.loc[:,1]=labelencoder_x_1.fit_transform(x.iloc[:,1])
labelencoder_x_8=LabelEncoder()
x.loc[:,8]=labelencoder_x_8.fit_transform(x.iloc[:,8])
labelencoder_x_10=LabelEncoder()
x.loc[:,10]=labelencoder_x_10.fit_transform(x.iloc[:,10])
#and to represent the chest_bain at the same way we need OneHotEncoder because there are 4kinds(ATA,NAP,ASY,TA)
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[2])],remainder='passthrough')
x=ct.fit_transform(x)
#x[2] (0010)NAP,(0100)ATA,(1000)ASY,(0001)TA
#represent resting (normal, st,lvh)
#x[6] (0)lvh,(1)normal,(2)st
#represent exerciseAngina x[8]
# (yes or no) (0)no and (1)yes
#x[10] flat (1) up(2) down(0)
#x[1] 0 female 1 male

#training and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scalling
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#ANN implementation
classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 14, 
                     units = 8, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "relu", units = 14, 
                     kernel_initializer = "uniform"))
classifier.add(Dense(activation = "sigmoid", units = 1, 
                     kernel_initializer = "uniform"))

classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', 
                   metrics = ['accuracy'] )
history=classifier.fit(x_train , y_train , batch_size = 100 ,epochs = 30,validation_data=(x_test,y_test)  )
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test,y_pred)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][1] + cm[1][0] +cm[0][0] +cm[1][1])

print('The accuracy *100')
print(accuracy*100)
print('The graph of accuracy is')
#graph of accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()

print('loss graph is')
#graph of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()

print('ROC graph is')
#graph of ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test,y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])
print(roc_auc_score(y_test,y_pred))
plt.figure()
plt.plot(fpr[1],tpr[1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.show()

print ('The confusion matrix is')
print(cm)
print('The graph of confusion matrix')
#graph of confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
prediction=np.round(classifier.predict(x_train))
cm = confusion_matrix(y_train, prediction)

cm_df = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[ ]:




