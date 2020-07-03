import io
import sys
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2

def multiply(a):
    x = np.multiply(a[0], a[1])
    print('x:',x)
    return x

class SeBlock(keras.layers.Layer):
  # class SeBlock(keras.layers.Layer):
  #     def __init__(self, channel, reduction_ratio=4,**kwargs):
  #         super(SeBlock,self).__init__(**kwargs)
  #         self.reduction_ratio = reduction_ratio
  #         self.channel = channel
  #         self.GlobalAveragePooling3D = keras.layers.GlobalAveragePooling3D(name='SE_Global_Average_Pooling')
  #         self.Dense_1 = keras.layers.Dense(self.channel / self.reduction_ratio, activation='linear', kernel_initializer='he_normal',
  #                     kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1a')
  #         self.Activation = keras.layers.Activation('relu', name='SE_ReLU_1')
  #         self.Dense_2 = keras.layers.Dense(self.channel, activation='linear', kernel_initializer='he_normal',
  #                     kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1b')
  #         self.sigmoid = keras.backend.sigmoid
  #         self.Reshape = keras.layers.Reshape([1,1,1,self.channel], name='SE_Reshape')
  #         self.Multiply = keras.layers.Multiply()
  #     def build(self,input_shape):
  #       # input_shape
  #       super(SeBlock, self).build(input_shape)
  #     def call(self, inputs):
  #         x = self.GlobalAveragePooling3D(inputs)
  #         x = self.Dense_1(x)
  #         x = self.Activation(x)
  #         x = self.Dense_2(x)
  #         x = self.sigmoid(x)
  #         x = self.Reshape(x)

  #         return self.Multiply([inputs,x])
    def __init__(self, channel, reduction_ratio=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.channel = channel
    def build(self,input_shape):
    	#input_shape
    	pass
    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling3D(name='SE_Global_Average_Pooling')(inputs)
        x = keras.layers.Dense(self.channel / self.reduction_ratio, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1a')(x)
        x = keras.layers.Activation('relu', name='SE_ReLU_1')(x)
        x = keras.layers.Dense(self.channel, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1b')(x)
        x = keras.backend.sigmoid(x)
        x = keras.layers.Reshape([1,1,1,self.channel], name='SE_Reshape')(x)
        return keras.layers.Multiply()([inputs,x])
        #return inputs*x

def SEnet(conv3d, channel, reduction_ratio, str_modality):

  Global_Average_Pooling = keras.layers.GlobalAveragePooling3D(name='SE_Global_Average_Pooling_%s'%str_modality)(conv3d)

  Fully_connected_a = keras.layers.Dense(channel / reduction_ratio, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(reduction_ratio), name='SE_Fully_connected_a_%s'%str_modality)(Global_Average_Pooling)

  ReLU_1 = keras.layers.Activation('relu', name='SE_ReLU_%s'%str_modality)(Fully_connected_a)

  Fully_connected_b = keras.layers.Dense(channel, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(reduction_ratio), name='SE_Fully_connected_b_%s'%str_modality)(ReLU_1)

  Sigmoid = keras.layers.Activation('sigmoid',name='SE_sigmoid_%s'%str_modality)(Fully_connected_b)
  excitation = keras.layers.Reshape([1,1,1,channel], name='SE_Reshape_%s'%str_modality)(Sigmoid)
  scale = keras.layers.Multiply(name='SE_Multiply_%s'%str_modality)([conv3d, excitation])

  return scale

def Res3D_Block1(inputs, weight_decay):
  # Res3D Block 1
  conv3d_1 = keras.layers.Conv3D(64, (3,7,7), strides=(1,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_1')(inputs)
  conv3d_1 = keras.layers.BatchNormalization(name='BatchNorm_1_0')(conv3d_1)
  conv3d_1 = keras.layers.Activation('relu', name='ReLU_1')(conv3d_1)

  # SEnet Block1
  conv3d_1 = SEnet(conv3d_1,64,4,'Block1')

  return conv3d_1

def Res3D_Block2(inputs, weight_decay):
  # Res3D Block 2
  conv3d_2a_1 = keras.layers.Conv3D(64, (1,1,1), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_1')(inputs)
  conv3d_2a_1 = keras.layers.BatchNormalization(name='BatchNorm_2a_1')(conv3d_2a_1)
  conv3d_2a_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_a')(inputs)
  conv3d_2a_a = keras.layers.BatchNormalization(name='BatchNorm_2a_a')(conv3d_2a_a)
  conv3d_2a_a = keras.layers.Activation('relu', name='ReLU_2a_a')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_b')(conv3d_2a_a)
  conv3d_2a_b = keras.layers.BatchNormalization(name='BatchNorm_2a_b')(conv3d_2a_b)
  conv3d_2a = keras.layers.Add(name='Add_2a')([conv3d_2a_1, conv3d_2a_b])
  conv3d_2a = keras.layers.Activation('relu', name='ReLU_2a')(conv3d_2a)

  conv3d_2b_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2b_a')(conv3d_2a)
  conv3d_2b_a = keras.layers.BatchNormalization(name='BatchNorm_2b_a')(conv3d_2b_a)
  conv3d_2b_a = keras.layers.Activation('relu', name='ReLU_2b_a')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2b_b')(conv3d_2b_a)
  conv3d_2b_b = keras.layers.BatchNormalization(name='BatchNorm_2b_b')(conv3d_2b_b)

  # SEnet Block2
  conv3d_2b_b = SEnet(conv3d_2b_b,64,4,'Block2')

  conv3d_2b = keras.layers.Add(name='Add_2b')([conv3d_2a, conv3d_2b_b])
  conv3d_2b = keras.layers.Activation('relu', name='ReLU_2b')(conv3d_2b)

  return conv3d_2b

def Res3D_Block3(inputs, weight_decay):
  # Res3D Block 3
  conv3d_3a_1 = keras.layers.Conv3D(128, (1,1,1), strides=(2,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_1')(inputs)
  conv3d_3a_1 = keras.layers.BatchNormalization(name='BatchNorm_3a_1')(conv3d_3a_1)
  conv3d_3a_a = keras.layers.Conv3D(128, (3,3,3), strides=(2,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_a')(inputs)
  conv3d_3a_a = keras.layers.BatchNormalization(name='BatchNorm_3a_a')(conv3d_3a_a)
  conv3d_3a_a = keras.layers.Activation('relu', name='ReLU_3a_a')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_b')(conv3d_3a_a)
  conv3d_3a_b = keras.layers.BatchNormalization(name='BatchNorm_3a_b')(conv3d_3a_b)
  conv3d_3a = keras.layers.Add(name='Add_3a')([conv3d_3a_1, conv3d_3a_b])
  conv3d_3a = keras.layers.Activation('relu', name='ReLU_3a')(conv3d_3a)

  conv3d_3b_a = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3b_a')(conv3d_3a)
  conv3d_3b_a = keras.layers.BatchNormalization(name='BatchNorm_3b_a')(conv3d_3b_a)
  conv3d_3b_a = keras.layers.Activation('relu', name='ReLU_3b_a')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3b_b')(conv3d_3b_a)
  conv3d_3b_b = keras.layers.BatchNormalization(name='BatchNorm_3b_b')(conv3d_3b_b)
  # SEnet Block3
  conv3d_3b_b = SEnet(conv3d_3b_b,128,4,'Block3')
  conv3d_3b = keras.layers.Add(name='Add_3b')([conv3d_3a, conv3d_3b_b])
  conv3d_3b = keras.layers.Activation('relu', name='ReLU_3b')(conv3d_3b)

  return conv3d_3b

def Res3D_Block4(inputs, weight_decay):
  # Res3D Block 4
  conv3d_4a_1 = keras.layers.Conv3D(256, (1,1,1), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_1')(inputs)
  conv3d_4a_1 = keras.layers.BatchNormalization(name='BatchNorm_4a_1')(conv3d_4a_1)
  conv3d_4a_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_a')(inputs)
  conv3d_4a_a = keras.layers.BatchNormalization(name='BatchNorm_4a_a')(conv3d_4a_a)
  conv3d_4a_a = keras.layers.Activation('relu', name='ReLU_4a_a')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_b')(conv3d_4a_a)
  conv3d_4a_b = keras.layers.BatchNormalization(name='BatchNorm_4a_b')(conv3d_4a_b)
  conv3d_4a = keras.layers.Add(name='Add_4a')([conv3d_4a_1, conv3d_4a_b])
  conv3d_4a = keras.layers.Activation('relu', name='ReLU_4a')(conv3d_4a)

  conv3d_4b_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4b_a')(conv3d_4a)
  conv3d_4b_a = keras.layers.BatchNormalization(name='BatchNorm_4b_a')(conv3d_4b_a)
  conv3d_4b_a = keras.layers.Activation('relu', name='ReLU_4b_a')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4b_b')(conv3d_4b_a)
  conv3d_4b_b = keras.layers.BatchNormalization(name='BatchNorm_4b_b')(conv3d_4b_b)

  # SEnet Block4
  conv3d_4b_b = SEnet(conv3d_4b_b,256,4,'Block4')

  conv3d_4b = keras.layers.Add(name='Add_4b')([conv3d_4a, conv3d_4b_b])
  conv3d_4b = keras.layers.Activation('relu', name='ReLU_4b')(conv3d_4b)

  return conv3d_4b

def res3d(inputs, weight_decay):
  res3d_1 = Res3D_Block1(inputs, weight_decay)  ### (2, 32, 56, 56, 64)
  res3d_2 = Res3D_Block2(res3d_1, weight_decay)
  res3d_3 = Res3D_Block3(res3d_2, weight_decay)
  res3d_4 = Res3D_Block4(res3d_3, weight_decay)

  return res3d_4


def relu6(x):
  return keras.activations.relu(x,max_value=6)

def mobilenet(inputs, weight_decay):
  conv2d_1a = keras.layers.SeparableConv2D(256, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1a')(inputs)
  conv2d_1a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1a')(conv2d_1a)
  conv2d_1a = keras.layers.Activation(relu6, name='ReLU_Conv2d_1a')(conv2d_1a)

  conv2d_1b = keras.layers.SeparableConv2D(256, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1b')(conv2d_1a)
  conv2d_1b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1b')(conv2d_1b)
  conv2d_1b = keras.layers.Activation(relu6, name='ReLU_Conv2d_1b')(conv2d_1b)

  conv2d_2a = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2a')(conv2d_1b)
  conv2d_2a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2a')(conv2d_2a)
  conv2d_2a = keras.layers.Activation(relu6, name='ReLU_Conv2d_2a')(conv2d_2a)

  conv2d_2b = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2b')(conv2d_2a)
  conv2d_2b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2b')(conv2d_2b)
  conv2d_2b = keras.layers.Activation(relu6, name='ReLU_Conv2d_2b')(conv2d_2b)

  conv2d_2c = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2c')(conv2d_2b)
  conv2d_2c = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2c')(conv2d_2c)
  conv2d_2c = keras.layers.Activation(relu6, name='ReLU_Conv2d_2c')(conv2d_2c)

  conv2d_2d = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2d')(conv2d_2c)
  conv2d_2d = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2d')(conv2d_2d)
  conv2d_2d = keras.layers.Activation(relu6, name='ReLU_Conv2d_2d')(conv2d_2d)

  conv2d_2e = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2e')(conv2d_2d)
  conv2d_2e = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2e')(conv2d_2e)
  conv2d_2e = keras.layers.Activation(relu6, name='ReLU_Conv2d_2e')(conv2d_2e)

  conv2d_3a = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3a')(conv2d_2e)
  conv2d_3a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3a')(conv2d_3a)
  conv2d_3a = keras.layers.Activation(relu6, name='ReLU_Conv2d_3a')(conv2d_3a)

  conv2d_3b = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3b')(conv2d_3a)
  conv2d_3b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3b')(conv2d_3b)
  conv2d_3b = keras.layers.Activation(relu6, name='ReLU_Conv2d_3b')(conv2d_3b)

  return conv2d_3b

def res3d_clstm_mobilenet(inputs, seq_len, weight_decay):
  # Res3D Block
  res3d_featmap = res3d(inputs, weight_decay)

  # GatedConvLSTM2D Block
  clstm2d_1 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_1')(res3d_featmap)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_2')(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape')(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape')(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4),
                    padding='valid', name='Average_Pooling')(features)

  return gpooling
