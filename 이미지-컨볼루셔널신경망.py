
# %%
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets, models, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# %%
# MNIST 데이터셋 다운로드하고 준비하기
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# %%
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

plt.imshow(train_images[0])
plt.show()

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# %%
# 합성곱 층 만들기
def build_model():
    model = models.Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# %%
# 모델 컴파일과 훈련하기
early = callbacks.EarlyStopping(monitor='val_loss', patience=10)
model.fit(train_images, train_labels, epochs=20, verbose=1, 
          validation_split=0.2, callbacks=[early], batch_size=512)

# %%
# 모델 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print(test_acc)

# %%
pred = model.predict(test_images).argmax(axis=1)

# %%
num = 22
plt.imshow(test_images[num])
plt.show()
print("pred: {:1.0f}, test: {:1.0f}".format(pred[num], test_labels[num]))

# %%
