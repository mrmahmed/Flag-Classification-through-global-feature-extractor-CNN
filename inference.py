from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix

classes=os.listdir(r'C:\Users\DR. WAQAR\Downloads\avivco97-attachments\result')


from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator()
#         rotation_range=90, # rotation
#         rescale=1./255,
#         width_shift_range=0.2, # horizontal shift
#         height_shift_range=0.2, # vertical shift
#         zoom_range=0.2, # zoom
#         horizontal_flip=True, # horizontal flip
# brightness_range=[0.2,1.2]) # brightness

data_generator = datagen.flow_from_directory(
                  directory=r'C:\Users\DR. WAQAR\Downloads\avivco97-attachments\result',
                  target_size=(227, 227), # resize to this size
                  color_mode="rgb", # for coloured images
                  batch_size=207, # number of images to extract from folder for every batch
                  class_mode="categorical", # classes to predict
                  seed=2020 # to make the result reproducible
)



def baseline_model():
    model = Sequential()
    # Conv1
    model.add(Conv2D(filters=14,kernel_size=21,strides=4,activation='relu',input_shape=(227,227,3)))
    # Max Pool 1
    model.add(MaxPool2D(pool_size=3, strides=2))
    # Conv2
    model.add(Conv2D(filters=7, kernel_size=11,strides=1,padding='valid', activation='relu' ))
    # Max Pool 2
    model.add(MaxPool2D(pool_size=3, strides=2 ))

    # Dropout
    # model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    # model.add(Dense(46,activation='relu'))
    model.add(Dense(207,activation='softmax'))
    return model



model = baseline_model()
model.load_weights('model_custom.h5')



pred_res=model.predict_generator(data_generator,steps=20907/207)

clas= np.argmax(pred_res,axis=1)


res=confusion_matrix(clas,data_generator.classes)


import seaborn as snNew
import pandas as pdNew
import matplotlib.pyplot as pltNew


DetaFrame_cm = pdNew.DataFrame(res, range(207), range(207))
snNew.heatmap(DetaFrame_cm, annot=True)
pltNew.show()


print(model.summary())

img=cv2.imread(r'C:\Users\DR. WAQAR\Downloads\avivco97-attachments\result\Guyana\Guyana0.png')

img=cv2.resize(img,(227,227))


img = img.astype('float32') / 255.

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img=np.expand_dims(img,0)

res=model.predict(img)

clas= np.argmax(res,axis=1)

print(classes[clas[0]])