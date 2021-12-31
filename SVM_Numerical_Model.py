import epochs as epochs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.models import sequential

mpl.use('TKAgg')
df = pd.read_csv('C:/Users/NAFADY/PycharmProjects/frist/heart.csv')
df.info()

# 1-Data_preprocessing

# 1.1 import libraries


# 1.2 load  the dataset
dataset = pd.read_csv('C:/Users/NAFADY/PycharmProjects/frist/heart.csv')

# 1.3 split dataset into x,y
x = pd.DataFrame(dataset.iloc[:, 0:11].values)
y = pd.DataFrame(dataset.iloc[:, 11].values)

# 1.4 Encode categorical data


labelencoder_x_6 = LabelEncoder()
x.loc[:, 6] = labelencoder_x_6.fit_transform(x.iloc[:, 6])
labelencoder_x_1 = LabelEncoder()
x.loc[:, 1] = labelencoder_x_1.fit_transform(x.iloc[:, 1])
labelencoder_x_8 = LabelEncoder()
x.loc[:, 8] = labelencoder_x_8.fit_transform(x.iloc[:, 8])
labelencoder_x_10 = LabelEncoder()
x.loc[:, 10] = labelencoder_x_10.fit_transform(x.iloc[:, 10])
# and to represent the chest_bain at the same way we need OneHotEncoder because there are 4kinds(ATA,NAP,ASY,TA)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
x = ct.fit_transform(x)
# x[2] (0010)NAP,(0100)ATA,(1000)ASY,(0001)TA
# represent resting (normal, st,lvh)
# x[6] (0)lvh,(1)normal,(2)st
# represent exerciseAngina x[8]
# (yes or no) (0)no and (1)yes
# x[10] flat (1) up(2) down(0)
# x[1] 0 female 1 male

# 1.5 training and test


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 1.6 feature scalling


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(matplotlib.get_backend())

df.describe()
print(df.describe())

df.shape
print(df.shape)

df.head()
print(df.head())

print(df['Sex'].unique())
print(df['ChestPainType'].unique())
print(df['RestingECG'].unique())
print(df['ExerciseAngina'].unique())

print(df['ST_Slope'].unique())
fig = plt.figure()
df['Sex'].value_counts()
sns.countplot(data=df, x='Sex')
plt.show()

fig = plt.figure()
df['ChestPainType'].value_counts()
sns.countplot(data=df, x='ChestPainType')
plt.show()

fig = plt.figure()
df['RestingECG'].value_counts()
sns.countplot(data=df, x='RestingECG')
plt.show()

fig = plt.figure()
df['ExerciseAngina'].value_counts()
sns.countplot(data=df, x='ExerciseAngina')
plt.show()

fig = plt.figure()
df['ST_Slope'].value_counts()
sns.countplot(data=df, x='ST_Slope')
plt.show()

df.isnull().sum()
df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
print(df)

df.corr()['HeartDisease'].sort_values()
print(df.corr()['HeartDisease'].sort_values())

plt.figure(figsize=(16, 9))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# checking for outliers
plt.figure(figsize=(20, 15))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(data=df, x='HeartDisease', y=df[df.columns[i]])
plt.show()

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

scaler = StandardScaler()
scaler.fit(X_train)
print(scaler.fit(X_train))

scaler_train = scaler.transform(X_train)
scaler_test = scaler.transform(X_test)

svc = SVC()
svc.fit(scaler_train, y_train)
y_pred = svc.predict(scaler_test)

pd.DataFrame({'Y test': y_test, 'Y predict': y_pred})
print(pd.DataFrame({'Y test': y_test, 'Y predict': y_pred}))

print('------------------------------------------------')

print('Accuracy Score Is: ')
accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

print('------------------------------------------------')

print('Confusion Matrix is:')

confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

print('------------------------------------------------')

print(classification_report(y_test, y_pred))

print('------------------------------------------------')

# svm = SVC()
# param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
# grid = GridSearchCV(svm, param_grid, cv=5)

# grid.fit(scaler_train, y_train)
# print(grid.fit(scaler_train, y_train))

print('------------------------------------------------')

# y_pred_grid = grid.predict(scaler_test)

# accuracy_score(y_test, y_pred_grid)
# print(accuracy_score(y_test, y_pred_grid))

# confusion_matrix(y_test, y_pred_grid)
# print(confusion_matrix)

# print(classification_report(y_test, y_pred_grid))

# roc curve
X = np.array(df.iloc[:, :11])
Y = np.array(df.iloc[:, -1])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf_RF = RandomForestClassifier(max_depth=5, n_estimators=2000, oob_score=True)
clf_RF.fit(X_train, y_train)
print("Accuracy:", clf_RF.oob_score_)
LGR = LogisticRegression(solver='liblinear', C=1)
LGR.fit(X_train, y_train)

clf_SVM = SVC(C=100, kernel='rbf')
clf_SVM.fit(X_train, y_train)

fig = plot_roc_curve(clf_SVM, X_test, y_test)
fig = plot_roc_curve(LGR, X_test, y_test, ax=fig.ax_)
fig = plot_roc_curve(clf_RF, X_test, y_test, ax=fig.ax_)
plt.title("ROC CURVE")
plt.show()

# confusion matrix
sns.heatmap(df[["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]].corr(), annot=True)
plt.title("Confusion Matrix")
plt.show()

# graph of loss
df = pd.read_csv('C:/Users/NAFADY/PycharmProjects/frist/heart.csv')
df.dropna(inplace=True)


def replace_column_with_one_hot(column, df):
    one_hot = pd.get_dummies(df[column])
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df


columns_to_replace = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for colname in columns_to_replace:
    df = replace_column_with_one_hot(colname, df)
train, test = train_test_split(df, test_size=0.2)

label = 'HeartDisease'

y_train = train.pop(label)
y_test = test.pop(label)

x_train = train
x_test = test
norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(x_train)
norm(x_train.iloc[:3])
model = tf.keras.Sequential([
  norm,
  tf.keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(train.shape[1],)),
  tf.keras.layers.Dense(8, activation=tf.nn.relu),
  tf.keras.layers.Dense(1),
])
learning_rate = 0.0001
epochs = 400

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric = tf.keras.metrics.BinaryCrossentropy(from_logits=False)

model.compile(optimizer=opt, loss=loss, metrics=[metric, 'accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, verbose=0)
print(history.history.keys())

plt.plot(history.history['loss'], c='r')
plt.plot(history.history['val_loss'], c='b')
plt.legend(['loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'], c='b')
plt.plot(history.history['val_accuracy'], c='r')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()
