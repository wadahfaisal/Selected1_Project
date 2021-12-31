import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
import glob as gb
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

dataset = "C:/Users/hp/OneDrive/Documents/FCAIH/level 3/selected/sel2/data/"

for folder in  os.listdir(dataset + '') : 
    files = gb.glob(pathname= str( dataset +'' + folder + '/*.png'))
    print(f'For training data , found {len(files)} in folder {folder}')

code = {'non-vehicles':0 ,'vehicles':1}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x 

size = []
for folder in  os.listdir(dataset +'') : 
    files = gb.glob(pathname= str( dataset +'' + folder + '/*.png'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

s=100

X_train = []
y_train = []
for folder in  os.listdir(dataset +'') : 
    files = gb.glob(pathname= str( dataset +'' + folder + '/*.png'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])
        

print(f'we have {len(X_train)} items in X_train')

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))
 
X_train = np.array(X_train)
y_train = np.array(y_train)

print(f'X_train shape  is {X_train.shape}')
print(f'y_train shape  is {y_train.shape}')

plt.style.use("ggplot")
plt.figure(figsize=(8,7))
sns.countplot(x = y_train)
plt.show()

le = LabelEncoder()
y_train= le.fit_transform(y_train)

label = to_categorical(y_train)
print(y_train.shape)
X_train,y_train = shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,random_state = 42)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer ="adam", loss = "binary_crossentropy", metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=10,batch_size=64,verbose=1, validation_data = (x_test,y_test))

model.evaluate(x_test,y_test)
#graph of accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()

#graph of loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()

#graph of confusion matrix

prediction=np.round(model.predict(x_train))
cm = confusion_matrix(y_train, prediction)

cm_df = pd.DataFrame(cm, index = [i for i in ['Non Vehicles', 'Vehicles']], columns = [i for i in ['Non Vehicles', 'Vehicles']])

plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
