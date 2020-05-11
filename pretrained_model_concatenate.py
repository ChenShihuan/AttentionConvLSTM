import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from SEres3d_clstm_mobilenet_Fusion import res3d_clstm_mobilenet, FusionRes3d_clstm_mobilenet
from callbacks import LearningRateScheduler 
from datagen import isoTrainImageGenerator, isoTestImageGenerator
from datagen import jesterTrainImageGenerator, jesterTestImageGenerator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model

# Modality
RGB = 0
Depth = 1
Flow = 2
Fusion = 3

# Dataset
JESTER = 0
ISOGD = 1

cfg_modality = Fusion
cfg_dataset = ISOGD

if cfg_modality==RGB:
  str_modality = 'rgb'
elif cfg_modality==Depth:
  str_modality = 'depth'
elif cfg_modality==Flow:
  str_modality = 'flow'
elif cfg_modality==Fusion:
  str_modality = 'fusion'

if cfg_dataset==JESTER:
  nb_epoch = 30
  init_epoch = 0
  seq_len = 16
  batch_size = 16
  num_classes = 27
  dataset_name = 'jester_%s'%str_modality
  training_datalist = './dataset_splits/Jester/train_%s_list.txt'%str_modality
  testing_datalist = './dataset_splits/Jester/valid_%s_list.txt'%str_modality
elif cfg_dataset==ISOGD:
  nb_epoch = 10
  init_epoch = 0
  seq_len = 32
  batch_size = 2
  num_classes = 249
  dataset_name = 'isogr_%s'%str_modality
  training_datalist = './dataset_splits/IsoGD/train_%s_list.txt'%str_modality
  testing_datalist = './dataset_splits/IsoGD/valid_%s_list.txt'%str_modality

weight_decay = 0.00005
model_prefix = './models/'

# load pretrained model
inputs_RGB = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))
inputs_Flow = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))

feature = FusionRes3d_clstm_mobilenet(inputs_RGB, inputs_Flow, seq_len, weight_decay, str_modality)
# feature_Flow = res3d_clstm_mobilenet(inputs_Flow, seq_len, weight_decay, 'Flow')

# x = keras.layers.Concatenate(axis=-1)([feature_RGB, feature_Flow])
# x = keras.layers.AveragePooling3D(name='GlobalAveragePooling1D')(x)

flatten = keras.layers.Flatten(name='Flatten_%s'%str_modality)(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)

model_Fusion  = keras.models.Model(inputs=[inputs_RGB, inputs_Flow], outputs=outputs)
# model_Fusion  = keras.models.Model(inputs=inputs_RGB, outputs=outputs)
print(model_Fusion.summary())
plot_model(model_Fusion,to_file="RewriteSEres3d_clstm_mobilenet_Fusion.png",show_shapes=True)
# model_IsoGD_FLow = keras.models.Model(inputs=inputs, outputs=feature)

# IsoGD_RGB_pretrained_model  = '%sisogr_rgb_gatedclstm_weights.h5'%(model_prefix)
# print 'Loading pretrained model from %s' % model_IsoGD_RGB
# IsoGD_FLow_pretrained_model = '%sisogr_flow_gatedclstm_weights.h5'%(model_prefix)
# print 'Loading pretrained model from %s' % model_IsoGD_FLow

# model_IsoGD_RGB.load_weights(IsoGD_RGB_pretrained_model, by_name=True)
# print(model_IsoGD_RGB.summary())
# # model_IsoGD_RGB = keras.models.Model(inputs=model_IsoGD_RGB.input, outputs=model_IsoGD_RGB.get_layer('Average_Pooling').output)
# # print(model_IsoGD_RGB.summary())

# model_IsoGD_FLow.load_weights(IsoGD_FLow_pretrained_model, by_name=True)
# # model_IsoGD_FLow = keras.models.Model(inputs=model_IsoGD_FLow.input, outputs=model_IsoGD_FLow.get_layer('Average_Pooling').output)
# print(model_IsoGD_FLow.summary())

# r1 = model_IsoGD_RGB.output
# r2 = model_IsoGD_FLow.output

# x = keras.layers.Concatenate(axis=-1)([r1, r2])

# flatten = keras.layers.Flatten(name='Flatten')(x)
# classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
#                     kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
# outputs = keras.layers.Activation('softmax', name='Output')(classes)
# model = keras.models.Model(inputs=inputs, outputs=outputs)
# print(model.summary())

