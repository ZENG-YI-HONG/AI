
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v1 import SGD
from keras.optimizer_v1 import Adam
from keras.optimizer_v1 import RMSprop
from tensorflow import optimizers


#設定xor的特徵
x1 = np.array([[0],[0],[1],[1]])
x2 = np.array([[0],[1],[1],[0]])   
y = np.array([[0],[1],[0],[1]])


features = tf.compat.v1.concat([x1, x2], 1)

#設定模組
model = Sequential()

# 設定輸入層隱藏層
model.add(Dense(units=4, input_shape=(2,), kernel_initializer='RandomNormal', activation='softmax'))

#輸出層
model.add(Dense(units=1, activation='sigmoid'))


#設定損失函式與優化器
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), metrics=['accuracy'])

#預損失值
print(model.summary())

#訓練模型
history = model.fit(features, y, epochs=500, batch_size=4, verbose=2)


predict_prob = model.predict(features)
print(predict_prob)
predict_classes = np.round(predict_prob)
print(predict_classes)

plt.figure(figsize=(8, 4))
plt.plot(np.linspace(0, 500, len(history.history['loss'])), history.history['loss'])
plt.xticks(np.linspace(0, 500, 11))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
