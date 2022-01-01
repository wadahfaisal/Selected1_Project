import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import cv2


path=os.listdir("C:/Users/WADAH/Documents/Datasets")
classes={'non-vehicles':0,'vehicles':1}



X=[]
y=[]
for cls in classes:
    pth='C:/Users/WADAH/Documents/Datasets/data/' + cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)  
        img = cv2.resize(img,(70,70))
        X.append(img)
        y.append(classes[cls])


np.unique(y)


X = np.array(X)
y = np.array(y)

pd.Series(y).value_counts()

plt.imshow(X[0],cmap='gray')

X_updated=X.reshape(len(X),-1)
X_updated.shape

xtrain,xtest,ytrain,ytest=train_test_split(X_updated,y,random_state=10,test_size=0.20)


xtrain.shape,xtest.shape



#-----------------Feature Scaling--------------------
#print(xtrain.max(), xtrain.min())
#print(xtest.max(), xtest.min())
xtrain=xtrain/255
xtest=xtest/255
#print(xtrain.max(), xtrain.min())
#print(xtest.max(), xtest.min())


  #-------------Feature Selection(PCA)-----------------

#print(xtrain.shape,xtest.shape)

pca=PCA(.98)
#pca_train=pca.fit_transform(xtrain)
#pca_test=pca.transform(xtest)
pca_train=xtrain
pca_test=xtest

#--------------Train Model---------------
from sklearn.linear_model import LogisticRegression
LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')
lg= LogisticRegression(C=0.1)
lg.fit(pca_train,ytrain)
sv=SVC()
sv.fit(pca_train, ytrain)


y_pred_svm = sv.decision_function(xtest)
y_pred_logistic = lg.decision_function(xtest)
#------------Evaluation-----------------
#print("Training Score:",lg.score(pca_train,ytrain))
#print("Testing Score:",lg.score(pca_test,ytest))

#print("Training Score:",sv.score(pca_train,ytrain))
#print("Testing Score:",sv.score(pca_test,ytest))

#----------------Prediction--------------------------
pred=sv.predict(xtest)  #y_pred_svm
# pred=sv.decision_function(xtest)
np.where(ytest!=pred)






classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)
#rint(np.concatenate((y_pred.reshape(len(y_pred),1), ytest.reshape(len(ytest),1)),1))


from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
cm = confusion_matrix(ytest,y_pred)
print("---------------------------------")
print("Confusion Matrix:",cm)
print("---------------------------------")
print("Accuracy: ",accuracy_score(ytest,pred)*100)


#----------------------------------------------------ROC-----------------------

logistic_fpr, logistic_tpr, threshold = roc_curve(ytest, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(ytest, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % auc_svm)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

plt.xlabel('False Positive Rate (1 - Sensitity)-->')
plt.ylabel('True Positive Rate (Sensitity)-->')

plt.legend()
print("--------------------------------------------------")
plt.title('ROC and AUC curves')

plt.show()


#-------------------------------Confusion Matrix Graph------------------------------------
import seaborn as sns

cm_df = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

#-----------------------------Learning Rate-------------------------------------------
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(classifier,
                                                        xtest, ytest, cv=10, scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.subplots(1, figsize=(6,6))
plt.plot(train_sizes, train_mean, '--', color='blue',  label="Training score")
plt.plot(train_sizes, test_mean, color='blue', label="Cross-validation score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
