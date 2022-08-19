#LOAD LIBRARIES

from keras.datasets import mnist
import matplotlib.pyplot as plt
# keras 2.2.4
#
from keras.datasets import mnist,cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils

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




#LOAD DATA AMD CONVERT SHAPE
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# num_pixels = X_train.shape[1] * X_train.shape[2]

#one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# print(y_train )
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

#CREATE MODEL
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
    # model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    # model.add(Dense(46,activation='relu'))
    model.add(Dense(207,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

model = baseline_model()
model.load_weights('model_custom.h5')
print(model.summary())
history=model.fit_generator(data_generator, epochs=3, steps_per_epoch=20700/207)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))



# model_json = model.to_json()
# with open("model_custom.json", "w") as json_file:
#     json_file.write(model_json)
model.save_weights("model_custom.h5")