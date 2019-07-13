from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K 
from keras.optimizers import SGD
import keras
import numpy as np

source_model = InceptionV3(weights='imagenet')
base_model = InceptionV3(weights='imagenet', include_top=False)

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x_train = np.random.random((1000, 224, 224, 3))
print(x_train.shape)
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x = base_model.output
x = GlobalAveragePooling2D()(x)
# 添加全连接层
x = Dense(1024, activation='relu')(x)
# 添加逻辑回归层
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.fit_generator(img, steps_per_epoch=10, epochs=50)
model.fit(x_train, y_train, batch_size=32, epochs=10)

source_model_txt = []
# 输出层数及名字
for i, layer in enumerate(source_model):
    source_model_txt.append((i, layer.name))
temp = []
for i, layer in enumerate(base_model.layers):
    temp.append(layer.name)

with open("Inception_v3_model.txt", 'w') as f:
    for i, layer in source_model_txt:
        f.write(str(str(i) + ' ' + layerold + '\n'))
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# model.fit_generator(img)


