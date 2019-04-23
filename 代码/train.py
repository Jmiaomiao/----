#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt

batch_size = 32 # 训练时每个批次的样本数

num_classes = 3 # 3类别

epochs = 20 # 训练20周期

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mv_keras_face_trained_model.h5'

img_w = 150
img_h = 150

# LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# 创建history实例
history = LossHistory()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练样本初始化处理
train_generator = train_datagen.flow_from_directory(
    './data/mv/train', # 本例，提供100 x 3 = 300 个训练样本
    target_size=(img_w, img_h),  # 图片格式调整为 150x150
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')  # matt，多分类

validation_generator = test_datagen.flow_from_directory(
    './data/mv/validation',# 本例，提供30 x 3 = 90 个验证样本
    target_size=(img_w, img_h),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical')  # matt，多分类

# 模型适配生成
model.fit_generator(
    train_generator, # 训练集
    samples_per_epoch=2400, # 训练集总样本数，如果提供样本数量不够，则调整图片（翻转、平移等）补足数量（本例，该函数补充2400-300个样本）
    nb_epoch=epochs,
    validation_data=validation_generator, # 验证集
    nb_val_samples=800, # 同上
    callbacks=[history]) # 回调函数，绘制批次（epoch）和精确度（acc）关系图表函数

# Save model and weights
if not os.path.isdir(save_dir): # 没有save_dir对应目录则建立
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# 显示批次（epoch）和精确度（acc）关系图表
history.loss_plot('epoch')
