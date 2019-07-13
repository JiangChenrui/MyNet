from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

model = MobileNet(weights='imagenet')

img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 将网络结构写入txt文件
model_w = []
for i, layer in enumerate(model.layers):
    print(i, layer.name)
    model_w.append((i, layer.name))
model_w = np.array(model_w)
with open("mobilenet_model.txt", 'w') as f:
    for i, layer in model_w:
        f.write(str(str(i) + ' ' + layer + '\n'))

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])