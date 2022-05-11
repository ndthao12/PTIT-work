from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import imageio
import os
 
from keras.models import load_model
 
model = load_model('model_classCar_2_saved.h5')

categories= ["astilbe", "bellflower", "black-eyed susan", "calendula", "california poppy", "tulip"]

tmp= 0
correct= 0
path= 'test_data\\bellflower'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 1: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('bellflower: ', correct)

correct= 0
tmp= 0
path= 'test_data\\astilbe'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 0: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('astilbe: ', correct)
correct= 0

tmp= 0
path= 'test_data\\black-eyed susan'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 2: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('black-eyed susan: ',correct)
correct= 0

tmp= 0
path= 'test_data\\calendula'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 3: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('calendula: ',correct)

correct= 0
tmp= 0
path= 'test_data\\california poppy'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 4: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('california poppy: ',correct)

correct= 0
tmp= 0
path= 'test_data\\tulip'
for filename in os.listdir(path):
    image_path= os.path.join(path, filename)   
    image= load_img(image_path)
    img = np.array(image)
    img= img / 255.0
    img= img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    if np.argmax(label[0])== 5: 
        correct+= 1
    tmp+= 1
    if tmp==100:
        break
print('tulip: ',correct)

# image = load_img('test_data\\astilbe/19811535_82ec45117e_c.jpg', target_size=(256, 256))



# img = img / 255.0

# img = img.reshape(1,256,256,3)


# label = model.predict(img)
# classify= np.argmax(label,axis=1)
# print("Predicted Class (0 - Cars , 1- Planes): ", label)