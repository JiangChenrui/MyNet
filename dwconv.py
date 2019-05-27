import keras
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
# 使用gpu
# import theano
# theano.config.device = 'gpu0'
# theano.config.floatX = 'float32'

# 构造数据
data = np.random.random((2000, 20))
labels = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

print(data)
# 返回张量
inputs = Input(shape=(784,))
print(inputs.shape)
# 定义层
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 定义模型
model = Model(input=inputs, ouput=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels)