'''
기본 분류: 의류 이미지 분류
'''

# %%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %%
# 패션 MNIST 데이터셋 임포트하기
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(train_images[0])
plt.show()

# %%
# 데이터 전처리
train_images = train_images / 255.0
test_images = test_images / 255.0

# %%
plt.figure(figsize=(7,7))
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.xticks([])
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# %%
# 모델 구성

# 1. 모델링
def build_model():
    model = keras.Sequential()
    
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    return model

model = build_model()
model.summary()

# %%
# 2. 모델 훈련
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(train_images, train_labels, epochs=20, validation_split=0.2,
          verbose=1, callbacks=[earlystop], batch_size=256)

# 3. 정확도 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# %%
# 4. 예측
pred = model.predict(test_images).argmax(axis=1)

# %%
num = 100

plt.imshow(test_images[num])
plt.xlabel("pred: {}, test_labels: {}".format(class_names[pred[num]], class_names[test_labels[num]]))
plt.show()

# %%
