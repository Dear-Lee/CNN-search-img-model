import re
import h5py
from PIL import Image
import os
import numpy as np
import keras

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model

from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils

width_scale = 128  # 缩放尺寸宽度
height_scale = 128  # 缩放尺寸高度
filepath = 'C:/Users/L/Desktop/PYthon设计/toStu图像分类与检索/data/train'#训练集路径
text_path = 'C:/Users/L/Desktop/PYthon设计/toStu图像分类与检索/data/test'#测试集路径
write_path = 'C:/Users/L/Desktop/PYthon设计/save'  # 训练集转为单通道图像写入的路径
write_path1 = 'C:/Users/L/Desktop/PYthon设计/save-text'# 测试集转为单通道图像写入的路径


# 遍历每一张图片进行处理
def eachFile(filepath, savepath):
    pathDir = os.listdir(filepath)
    imgNum = len(pathDir) - 1
    for i in range(imgNum):
        image = Image.open(filepath + "/" + pathDir[i]).convert('L')  # 调整为单通道图像
        outimage = image.resize((width_scale, height_scale), Image.ANTIALIAS)  # 调整尺寸为128x128大小
        outimage.save(savepath + "/" + pathDir[i])
    # image.show(outimage)


if __name__ == '__main__':
    eachFile(filepath, write_path)
    eachFile(text_path, write_path1)


def data_label(path):
    pathDir = os.listdir(path)
    imgNum = len(pathDir) - 1
    #print(imgNum)
    data = np.empty((imgNum, 1, 128, 128), dtype='float32')  # 建立空的四维张量类型32位浮点
    label = np.empty((imgNum,), dtype='uint8')
    i=0
    for each_image in range(imgNum):
        #  image = Image.open(filepath + "/" + pathDir[each_image])
        image = Image.open(path + "/" + pathDir[each_image])
        mul_num = re.findall(r"\d", path + "/" + pathDir[each_image])  # 寻找字符串中的数字
        num = int(mul_num[0]) - 3
       # print(mul_num)
        arr = np.asarray(image, dtype="float32")
        data[i, :, :, :] = arr
       # label = int(pathDir[i].split('.')[0])
        label[i] = int (num)
        # print(label)
        i=i+1
    return data, label

#建立CNN模型
def cnn_model(train_data, train_label, test_data, test_label):

    model = keras.Sequential()    #生成一个model
  # model = load_model('C:/Users/L/Desktop/PYthon设计/my_model.h5')
    model.add(Convolution2D(
        nb_filter=12,
        nb_row=3,
        nb_col=3,
        border_mode='valid',
        dim_ordering='th',
        input_shape=(1, 128, 128)))
    model.add(Activation('relu'))  # 激活函数使用修正线性单元
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    model.add(Convolution2D(
        24,
        3,
        3,
        border_mode='valid',
        dim_ordering='th'))
    model.add(Activation('relu'))
    # 池化层 24×29×29
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    model.add(Convolution2D(
        48,
        3,
        3,
        border_mode='valid',
        dim_ordering='th'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.4))
    model.add(Dense(5, init='normal'))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
  #  model.save('C:/Users/L/Desktop/PYthon设计/my_model.h5')有问题
   # del model
    print('----------------training-----------------------')
    model.fit(train_data, train_label, batch_size=20, nb_epoch=50, shuffle=True  )# show_accuracy=True,
              #validation_split=0.1)
    print('----------------testing------------------------')
    loss, accuracy = model.evaluate(test_data, test_label)
    print('\n test loss:', loss)
    print('\n test accuracy', accuracy)



# train_path = write_path #filepath
# test_path1 = write_path1 #'C:/Users/L/Desktop/PYthon设计/toStu图像分类与检索/data/test'
train_data, train_label = data_label(write_path)
test_data, test_label = data_label(write_path1)
train_label = np_utils.to_categorical(train_label, num_classes=5)
test_label = np_utils.to_categorical(test_label, num_classes=5)
cnn_model(train_data, train_label, test_data, test_label)

