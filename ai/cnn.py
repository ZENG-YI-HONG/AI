from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

(train_image,train_label),(test_image,test_label)=mnist.load_data()

#資料初始化
train_image = train_image.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_image = test_image.reshape(10000, 28, 28, 1).astype('float')/255
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

print(tf.test.is_gpu_available())
print('tran_x', train_image.shape)
print('test_x',test_image.shape)


#建立cnn model
model = Sequential()
model.add(Dense(units=64,  activation='relu', kernel_initializer='normal'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu', kernel_initializer='normal' ))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn = model.fit(train_image, train_label , epochs=5, batch_size=64, validation_data=(test_image, test_label))



#lenet
lenet = Sequential()
lenet.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28,28,1)))
lenet.add(AveragePooling2D(pool_size=(2, 2), padding='same', strides=2))
lenet.add(Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
lenet.add(AveragePooling2D(pool_size=(2, 2), padding='same', strides=2))
lenet.add(Flatten())
lenet.add(Dense(units=100, activation='relu'))
lenet.add(Dropout(0.5))
lenet.add(Dense(units=60, activation='relu'))
lenet.add(Dropout(0.5))
lenet.add(Dense(units=10, activation='softmax'))
lenet.summary()

lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lenet.fit(test_image, test_label, epochs=5, batch_size=64)

"""""""""
plt.plot(cnn.history['accuracy'], label='CNN Training Accuracy')
plt.plot(lenet.history['accuracy'], label='LeNet Training Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""""""""
# 使用 CNN 模型預測
cnn_predictions = model.predict(test_image)
cnn_true_labels = np.argmax(test_label, axis=1)
cnn_predicted_labels = np.argmax(cnn_predictions, axis=1)

# 使用 LeNet 模型預測
lenet_predictions = lenet.predict(test_image)
lenet_true_labels = np.argmax(test_label, axis=1)
lenet_predicted_labels = np.argmax(lenet_predictions, axis=1)

# 使用classification_report打印报告
print("CNN Classification Report:")
print(classification_report(cnn_true_labels, cnn_predicted_labels))

print("LeNet Classification Report:")
print(classification_report(lenet_true_labels, lenet_predicted_labels))
