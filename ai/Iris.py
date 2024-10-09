import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from keras.layers import Dense
from keras.optimizer_v1 import RMSprop


from tensorflow import optimizers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score


df = pd.read_csv("C:\w\Iris.csv")

model=Sequential()


size_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Species'] = df['Species'].map(size_mapping)

Iris_setosa = df[df['Species'] == 0]
Iris_versicolor = df[df['Species'] == 1]
Iris_virginica = df[df['Species'] == 2]

x = np.asarray(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
y = np.asarray(df[['Species']])



# 設定輸入與隱藏層第一層
model.add(Dense(units=256, input_shape=(4,), kernel_initializer='RandomNormal', activation='relu'))
# 第二隱藏層
model.add(Dense(units=256, kernel_initializer='RandomNormal', activation='relu'))

#輸出層
model.add(Dense(units=3, activation='softmax'))

#設定損失函式與優化器
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), metrics=['accuracy'])

#顯示目前網路架構
print(model.summary())

#訓練模型
history = model.fit(x, y, epochs=1000, batch_size=128, verbose=2)




train_X , test_X , train_y , test_y = train_test_split( x, y, test_size=0.2, random_state=3)
print ('訓練資料集:', train_X.shape,  train_y.shape)
print ('驗證資料集:', test_X.shape,  test_y.shape)
print(df.dtypes)


pred_y = model.predict(test_X).argmax(axis=1)
cm = confusion_matrix(test_y, pred_y, labels=[0,1,2])
print(cm)
np.set_printoptions(precision=2)
print (classification_report(test_y, pred_y))
print(f1_score(test_y, pred_y, average='weighted'))
print(jaccard_score(test_y, pred_y, average='weighted'))


