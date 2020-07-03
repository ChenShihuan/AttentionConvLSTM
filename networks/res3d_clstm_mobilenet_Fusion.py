import io
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
keras = tf.contrib.keras
l2 = keras.regularizers.l2


class CrossStitchBlock(keras.layers.Layer):
    def __init__(self, batch_size, channel, **kwargs):
        self.self_initializer = keras.initializers.RandomUniform(
            minval=0.95, maxval=1.05, seed=None)
        self.cross_initializer = keras.initializers.RandomUniform(
            minval=-0.05, maxval=0.05, seed=None)

        self.batch_size = batch_size
        self.kernel_regularizer=keras.regularizers.l2(0.)
        self.channel = channel
        super(CrossStitchBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        self.alphaAA = self.add_weight(name='alphaAA',
                                      shape=[self.batch_size,1,1,1,self.channel],
                                      initializer=self.self_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.alphaAB = self.add_weight(name='alphaAB',
                                      shape=[self.batch_size,1,1,1,self.channel],
                                      initializer=self.cross_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        super(CrossStitchBlock, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        xa, xb = inputs

        xAA = keras.layers.Multiply()([self.alphaAA,xa])
        xAB = keras.layers.Multiply()([self.alphaAB,xb])

        xa_estimated = keras.layers.Add(name='xa_estimated')([xAA, xAB])
        
        return xa_estimated

    # def compute_output_shape(self, input_shape):
    #     assert isinstance(input_shape, list)
    #     shape_a, shape_b = input_shape
    #     return [xa_estimated, xb_estimated]

class CrossBlock(keras.layers.Layer):
  # class Cross_U(nn.Module):
  #   def __init__(self):
  #       super(Cross_U, self).__init__()
  #       self.conv1 = nn.Conv1d(2, 1, 1, bias=False)
  #       self.conv2 = nn.Conv1d(2, 1, 1, bias=False)
  #       self.conv3 = nn.Conv1d(2, 1, 1, bias=False)

  #   def forward(self, x1, x2):
  #       s = x1.size()
  #       to1d = lambda x: x.view(s[0], -1, 1)
  #       conc = torch.cat([to1d(x1), to1d(x2)], dim = 2).permute(0, 2, 1)
  #       conc = self.conv1(conc).permute(0, 2, 1).view(s)

  #       output1 = torch.cat([to1d(x1), conc], dim = 2).permute(0, 2, 1)
  #       output1 = self.conv2(output1).permute(0, 2, 1).view(s)

  #       output2 = torch.cat([to1d(x2), conc], dim = 2).permute(0, 2, 1)
  #       output2 = self.conv3(output2).permute(0, 2, 1).view(s)
  #       return output1, output2


    # arr_K = tensor_split(inp_K)
    # arr_K = run_layer_on_arr(arr_K, self.conv1_K)
    # arr_K = run_layer_on_arr(arr_K, self.bn1_K)
    # arr_K = run_layer_on_arr(arr_K, self.relu_K)
    # arr_K = run_layer_on_arr(arr_K, self.maxpool_K)

    # arr_M = tensor_split(inp_M)
    # arr_M = run_layer_on_arr(arr_M, self.conv1_M)
    # arr_M = run_layer_on_arr(arr_M, self.bn1_M)
    # arr_M = run_layer_on_arr(arr_M, self.relu_M)
    # arr_M = run_layer_on_arr(arr_M, self.maxpool_M)

    # arr_K_M = [self.CU1(k, m) for k, m in zip(arr_K, arr_M)]
    # arr_K = [arr_K_M[i][0] for i in range(0, len(arr_K_M))]
    # arr_M = [arr_K_M[i][1] for i in range(0, len(arr_K_M))]
    # t_K = self.layer1_K(tensor_merge(arr_K))
    # t_M = self.layer1_M(tensor_merge(arr_M))
    # arr_K = tensor_split(t_K)
    # arr_M = tensor_split(t_M)

    # keras.layers.Permute

    # FusionBlock verion 2
    def __init__(self, weight_decay, **kwargs):
        super(CrossBlock,self).__init__(**kwargs)
        # self.reduction_ratio = reduction_ratio
        self.weight_decay = weight_decay
    def build(self,input_shape):
      # input_shape
      super(CrossBlock, self).build(input_shape)
    def call(self, inputs):
      # x1 means the self channel, and x2 means another
      x1 = inputs[0]
      x2 = inputs[1]
      inputs_shapes = keras.backend.int_shape(x1)
      # print('shape',inputs_shapes)

      x1 = keras.layers.Reshape([-1,1])(x1)
      x2 = keras.layers.Reshape([-1,1])(x2)

      # Fusion x1 and x2
      Conc = keras.layers.Concatenate(axis=2)([x1, x2])
      Conc = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)(Conc)

      # Fusion conc and x1
      Output = keras.layers.Concatenate(axis=2)([x1, Conc])
      Output = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)(Output)      

      Output = keras.layers.Reshape([inputs_shapes[1],inputs_shapes[2],inputs_shapes[3],-1])(Output)
      return Output

def CrossFuction(inputs_A,inputs_B):
  inputs_shapes = keras.backend.int_shape(inputs_A)

  inputs_A = keras.layers.Reshape([-1,1])(inputs_A)
  inputs_B = keras.layers.Reshape([-1,1])(inputs_B)

  Conc = keras.layers.Concatenate(axis=2)([inputs_A, inputs_B])
  Conc = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)(Conc)

  Output = keras.layers.Concatenate(axis=2)([x1, Conc])
  Output = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)(Output)      

  Output = keras.layers.Reshape([inputs_shapes[1],inputs_shapes[2],inputs_shapes[3],-1])(Output) 
  return Output

class CatConvBlock(keras.layers.Layer):
    # FusionBlock verion 1
    def __init__(self, weight_decay, **kwargs):
        super(CatConvBlock,self).__init__(**kwargs)
        # self.reduction_ratio = reduction_ratio
        self.weight_decay = weight_decay
    def build(self,input_shape):
      # input_shape
      super(CatConvBlock, self).build(input_shape)
    def call(self, inputs):
      x1 = inputs[0]
      x2 = inputs[1]

      inputs_shapes = keras.backend.int_shape(x1)
      # print('shape',inputs_shapes)

      x1 = keras.layers.Reshape([-1,1])(x1)
      x2 = keras.layers.Reshape([-1,1])(x2)

      x = keras.layers.Concatenate(axis=2)([x1, x2])

      x = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(self.weight_decay), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)(x)

      x = keras.layers.Reshape([inputs_shapes[1],inputs_shapes[2],inputs_shapes[3],-1])(x)
      return x

def CatConvFuction(inputs_A,inputs_B,weight_decay):

  inputs_shapes = keras.backend.int_shape(inputs_A)

  inputs_A = keras.layers.Reshape([-1,1])(inputs_A)
  inputs_B = keras.layers.Reshape([-1,1])(inputs_B)
  # print(inputs_A)
  x = keras.layers.Concatenate(axis=2)([inputs_A, inputs_B])
  # print(x)
  # kernel_shape = keras.backend.int_shape(x)
  # kernel_size = kernel_shape[1]

  # x = keras.layers.Conv1D(filters = 2, kernel_size = filters_size, strides=1, padding='same',
  #                         kernel_initializer='he_normal',
  #                         kernel_regularizer=l2(weight_decay), activity_regularizer=None,
  #                         activation='relu',kernel_constraint=None, use_bias=False)(x)
  # print(x)
  x = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay), activity_regularizer=None,
                          activation='relu',kernel_constraint=None, use_bias=False)(x)

  x = keras.layers.Reshape([inputs_shapes[1],inputs_shapes[2],inputs_shapes[3],-1])(x)  
  return x

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

class SeBlock(keras.layers.Layer):
  # class SeBlock(keras.layers.Layer):
  #     def __init__(self, channel, reduction_ratio=4,**kwargs):
  #         super(SeBlock,self).__init__(**kwargs)
  #         self.reduction_ratio = reduction_ratio
  #         self.channel = channel
  #         self.GlobalAveragePooling3D = keras.layers.GlobalAveragePooling3D()
  #         self.Dense_1 = keras.layers.Dense(self.channel / self.reduction_ratio, activation='linear', kernel_initializer='he_normal',
  #                     kernel_regularizer=l2(self.reduction_ratio))
  #         self.Activation = keras.layers.Activation('relu', name='SE_ReLU_1')
  #         self.Dense_2 = keras.layers.Dense(self.channel, activation='linear', kernel_initializer='he_normal',
  #                     kernel_regularizer=l2(self.reduction_ratio))
  #         self.sigmoid = keras.backend.sigmoid
  #         self.Reshape = keras.layers.Reshape([1,1,1,self.channel])
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
      # input_shape
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
        # return inputs*x

def Res3D_Block1(inputs, weight_decay, str_modality):

  # Res3D Block 1
  conv3d_1 = keras.layers.Conv3D(64, (3,7,7), strides=(1,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_1_%s'%str_modality)(inputs)
  conv3d_1 = keras.layers.BatchNormalization(name='BatchNorm_1_0_%s'%str_modality)(conv3d_1)
  conv3d_1 = keras.layers.Activation('relu', name='ReLU_1_%s'%str_modality)(conv3d_1)

  # SEnet Block1
  # conv3d_1 = SEnet(conv3d_1,64,4,'Block1_%s'%str_modality)

  return conv3d_1

def Res3D_Block2(inputs, weight_decay, str_modality):

  # Res3D Block 2
  conv3d_2a_1 = keras.layers.Conv3D(64, (1,1,1), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_1_%s'%str_modality)(inputs)
  conv3d_2a_1 = keras.layers.BatchNormalization(name='BatchNorm_2a_1_%s'%str_modality)(conv3d_2a_1)
  conv3d_2a_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_a_%s'%str_modality)(inputs)
  conv3d_2a_a = keras.layers.BatchNormalization(name='BatchNorm_2a_a_%s'%str_modality)(conv3d_2a_a)
  conv3d_2a_a = keras.layers.Activation('relu', name='ReLU_2a_a_%s'%str_modality)(conv3d_2a_a)
  conv3d_2a_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2a_b_%s'%str_modality)(conv3d_2a_a)
  conv3d_2a_b = keras.layers.BatchNormalization(name='BatchNorm_2a_b_%s'%str_modality)(conv3d_2a_b)
  conv3d_2a = keras.layers.Add(name='Add_2a_%s'%str_modality)([conv3d_2a_1, conv3d_2a_b])
  conv3d_2a = keras.layers.Activation('relu', name='ReLU_2a_%s'%str_modality)(conv3d_2a)

  conv3d_2b_a = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2b_a_%s'%str_modality)(conv3d_2a)
  conv3d_2b_a = keras.layers.BatchNormalization(name='BatchNorm_2b_a_%s'%str_modality)(conv3d_2b_a)
  conv3d_2b_a = keras.layers.Activation('relu', name='ReLU_2b_a_%s'%str_modality)(conv3d_2b_a)
  conv3d_2b_b = keras.layers.Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_2b_b_%s'%str_modality)(conv3d_2b_a)
  conv3d_2b_b = keras.layers.BatchNormalization(name='BatchNorm_2b_b_%s'%str_modality)(conv3d_2b_b)

  # SEnet Block2
  # conv3d_2b_b = SEnet(conv3d_2b_b,64,4,'Block2_%s'%str_modality)

  conv3d_2b = keras.layers.Add(name='Add_2b_%s'%str_modality)([conv3d_2a, conv3d_2b_b])
  conv3d_2b = keras.layers.Activation('relu', name='ReLU_2b_%s'%str_modality)(conv3d_2b)

  return conv3d_2b

def Res3D_Block3(inputs, weight_decay, str_modality):

  # Res3D Block 3
  conv3d_3a_1 = keras.layers.Conv3D(128, (1,1,1), strides=(2,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_1_%s'%str_modality)(inputs)
  conv3d_3a_1 = keras.layers.BatchNormalization(name='BatchNorm_3a_1_%s'%str_modality)(conv3d_3a_1)
  conv3d_3a_a = keras.layers.Conv3D(128, (3,3,3), strides=(2,2,2), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_a_%s'%str_modality)(inputs)
  conv3d_3a_a = keras.layers.BatchNormalization(name='BatchNorm_3a_a_%s'%str_modality)(conv3d_3a_a)
  conv3d_3a_a = keras.layers.Activation('relu', name='ReLU_3a_a_%s'%str_modality)(conv3d_3a_a)
  conv3d_3a_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3a_b_%s'%str_modality)(conv3d_3a_a)
  conv3d_3a_b = keras.layers.BatchNormalization(name='BatchNorm_3a_b_%s'%str_modality)(conv3d_3a_b)
  conv3d_3a = keras.layers.Add(name='Add_3a_%s'%str_modality)([conv3d_3a_1, conv3d_3a_b])
  conv3d_3a = keras.layers.Activation('relu', name='ReLU_3a_%s'%str_modality)(conv3d_3a)

  conv3d_3b_a = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3b_a_%s'%str_modality)(conv3d_3a)
  conv3d_3b_a = keras.layers.BatchNormalization(name='BatchNorm_3b_a_%s'%str_modality)(conv3d_3b_a)
  conv3d_3b_a = keras.layers.Activation('relu', name='ReLU_3b_a_%s'%str_modality)(conv3d_3b_a)
  conv3d_3b_b = keras.layers.Conv3D(128, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_3b_b_%s'%str_modality)(conv3d_3b_a)
  conv3d_3b_b = keras.layers.BatchNormalization(name='BatchNorm_3b_b_%s'%str_modality)(conv3d_3b_b)

  # SEnet Block3
  # conv3d_3b_b = SEnet(conv3d_3b_b,128,4,'Block3_%s'%str_modality)

  conv3d_3b = keras.layers.Add(name='Add_3b_%s'%str_modality)([conv3d_3a, conv3d_3b_b])
  conv3d_3b = keras.layers.Activation('relu', name='ReLU_3b_%s'%str_modality)(conv3d_3b)

  return conv3d_3b

def Res3D_Block4(inputs, weight_decay, str_modality):

  # Res3D Block 4
  conv3d_4a_1 = keras.layers.Conv3D(256, (1,1,1), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_1_%s'%str_modality)(inputs)
  conv3d_4a_1 = keras.layers.BatchNormalization(name='BatchNorm_4a_1_%s'%str_modality)(conv3d_4a_1)
  conv3d_4a_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_a_%s'%str_modality)(inputs)
  conv3d_4a_a = keras.layers.BatchNormalization(name='BatchNorm_4a_a_%s'%str_modality)(conv3d_4a_a)
  conv3d_4a_a = keras.layers.Activation('relu', name='ReLU_4a_a_%s'%str_modality)(conv3d_4a_a)
  conv3d_4a_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4a_b_%s'%str_modality)(conv3d_4a_a)
  conv3d_4a_b = keras.layers.BatchNormalization(name='BatchNorm_4a_b_%s'%str_modality)(conv3d_4a_b)
  conv3d_4a = keras.layers.Add(name='Add_4a_%s'%str_modality)([conv3d_4a_1, conv3d_4a_b])
  conv3d_4a = keras.layers.Activation('relu', name='ReLU_4a_%s'%str_modality)(conv3d_4a)

  conv3d_4b_a = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4b_a_%s'%str_modality)(conv3d_4a)
  conv3d_4b_a = keras.layers.BatchNormalization(name='BatchNorm_4b_a_%s'%str_modality)(conv3d_4b_a)
  conv3d_4b_a = keras.layers.Activation('relu', name='ReLU_4b_a_%s'%str_modality)(conv3d_4b_a)
  conv3d_4b_b = keras.layers.Conv3D(256, (3,3,3), strides=(1,1,1), padding='same',
                    dilation_rate=(1,1,1), kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), use_bias=False,
                    name='Conv3D_4b_b_%s'%str_modality)(conv3d_4b_a)
  conv3d_4b_b = keras.layers.BatchNormalization(name='BatchNorm_4b_b_%s'%str_modality)(conv3d_4b_b)
  
  # SEnet Block4
  # conv3d_4b_b = SEnet(conv3d_4b_b,256,4,'Block4_%s'%str_modality)

  conv3d_4b = keras.layers.Add(name='Add_4b_%s'%str_modality)([conv3d_4a, conv3d_4b_b])
  conv3d_4b = keras.layers.Activation('relu', name='ReLU_4b_%s'%str_modality)(conv3d_4b)

  return conv3d_4b

def res3d(inputs, weight_decay, str_modality):
  res3d_1 = Res3D_Block1(inputs, weight_decay, str_modality)  ### (2, 32, 56, 56, 64)
  res3d_2 = Res3D_Block2(res3d_1, weight_decay, str_modality)
  res3d_3 = Res3D_Block3(res3d_2, weight_decay, str_modality)
  res3d_4 = Res3D_Block4(res3d_3, weight_decay, str_modality)

  return res3d_4

def CatConvFusionRes3d(inputs_RGB, inputs_Flow, weight_decay):
  # FusionRes3d verion 1
  res3d_1_RGB = Res3D_Block1(inputs_RGB, weight_decay, 'rgb')
  res3d_1_Flow = Res3D_Block1(inputs_Flow, weight_decay, 'flow')
  res3d_1 = CatConvBlock(weight_decay)([res3d_1_RGB, res3d_1_Flow])

  res3d_2_RGB = Res3D_Block2(res3d_1, weight_decay, 'rgb')
  res3d_2_Flow = Res3D_Block2(res3d_1, weight_decay, 'flow')
  res3d_2 = CatConvBlock(weight_decay)([res3d_2_RGB, res3d_2_Flow])

  res3d_3_RGB = Res3D_Block3(res3d_2, weight_decay, 'rgb')
  res3d_3_Flow = Res3D_Block3(res3d_2, weight_decay, 'flow')
  res3d_3 = CatConvBlock(weight_decay)([res3d_3_RGB, res3d_3_Flow])

  res3d_4_RGB = Res3D_Block4(res3d_3, weight_decay, 'rgb')
  res3d_4_Flow = Res3D_Block4(res3d_3, weight_decay, 'flow')
  res3d_4 = CatConvBlock(weight_decay)([res3d_4_RGB, res3d_4_Flow])
  return res3d_4

def FusionRes3d(inputs_RGB, inputs_Flow, weight_decay):
  # FusionRes3d verion 2
  res3d_1_RGB = Res3D_Block1(inputs_RGB, weight_decay, 'rgb')
  res3d_1_Flow = Res3D_Block1(inputs_Flow, weight_decay, 'flow')
  Cross_1_RGB = CrossBlock(weight_decay,name='Cross_1_RGB')([res3d_1_RGB, res3d_1_Flow])
  Cross_1_Flow = CrossBlock(weight_decay,name='Cross_1_Flow')([res3d_1_Flow, res3d_1_RGB])

  res3d_2_RGB = Res3D_Block2(Cross_1_RGB, weight_decay, 'rgb')
  res3d_2_Flow = Res3D_Block2(Cross_1_Flow, weight_decay, 'flow')
  Cross_2_RGB = CrossBlock(weight_decay,name='Cross_2_RGB')([res3d_2_RGB, res3d_2_Flow])
  Cross_2_Flow = CrossBlock(weight_decay,name='Cross_2_Flow')([res3d_2_Flow, res3d_2_RGB])

  res3d_3_RGB = Res3D_Block3(Cross_2_RGB, weight_decay, 'rgb')
  res3d_3_Flow = Res3D_Block3(Cross_2_Flow, weight_decay, 'flow')
  Cross_3_RGB = CrossBlock(weight_decay,name='Cross_3_RGB')([res3d_3_RGB, res3d_3_Flow])
  Cross_3_Flow = CrossBlock(weight_decay,name='Cross_3_Flow')([res3d_3_Flow, res3d_3_RGB])

  res3d_4_RGB = Res3D_Block4(Cross_3_RGB, weight_decay, 'rgb')
  res3d_4_Flow = Res3D_Block4(Cross_3_Flow, weight_decay, 'flow')
  res3d_4 = CatConvBlock(weight_decay,name='CatConvBlock')([res3d_4_RGB, res3d_4_Flow])

  return res3d_4

def relu6(x):
  return keras.activations.relu(x,max_value=6)

def StitchRes3d(inputs_RGB, inputs_Flow, weight_decay):
  res3d_1_RGB = Res3D_Block1(inputs_RGB, weight_decay, 'rgb')
  res3d_1_Flow = Res3D_Block1(inputs_Flow, weight_decay, 'flow')

  Stitch_1_RGB = CrossStitchBlock(batch_size=2,channel=64,name='Stitch_1_RGB')([res3d_1_RGB, res3d_1_Flow])
  Stitch_1_Flow = CrossStitchBlock(batch_size=2,channel=64,name='Stitch_1_Flow')([res3d_1_Flow, res3d_1_RGB])

  res3d_2_RGB = Res3D_Block2(Stitch_1_RGB, weight_decay, 'rgb')
  res3d_2_Flow = Res3D_Block2(Stitch_1_Flow, weight_decay, 'flow')
  Stitch_2_RGB = CrossStitchBlock(batch_size=2,channel=64,name='Stitch_2_RGB')([res3d_2_RGB, res3d_2_Flow])
  Stitch_2_Flow = CrossStitchBlock(batch_size=2,channel=64,name='Stitch_2_Flow')([res3d_2_Flow, res3d_2_RGB])

  res3d_3_RGB = Res3D_Block3(Stitch_2_RGB, weight_decay, 'rgb')
  res3d_3_Flow = Res3D_Block3(Stitch_2_Flow, weight_decay, 'flow')
  Stitch_3_RGB = CrossStitchBlock(batch_size=2,channel=128,name='Stitch_3_RGB')([res3d_3_RGB, res3d_3_Flow])
  Stitch_3_Flow = CrossStitchBlock(batch_size=2,channel=128,name='Stitch_3_Flow')([res3d_3_Flow, res3d_3_RGB])

  res3d_4_RGB = Res3D_Block4(Stitch_3_RGB, weight_decay, 'rgb')
  res3d_4_Flow = Res3D_Block4(Stitch_3_Flow, weight_decay, 'flow')
  Stitch_4_RGB = CrossStitchBlock(batch_size=2,channel=256,name='Stitch_4_RGB')([res3d_4_RGB, res3d_4_Flow])
  Stitch_4_Flow = CrossStitchBlock(batch_size=2,channel=256,name='Stitch_4_Flow')([res3d_4_Flow, res3d_4_RGB])

  return Stitch_4_RGB, Stitch_4_Flow

def relu6(x):
  return keras.activations.relu(x,max_value=6)

def mobilenet(inputs, weight_decay, str_modality):

  conv2d_1a = keras.layers.SeparableConv2D(256, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1a_%s'%str_modality)(inputs)
  conv2d_1a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1a_%s'%str_modality)(conv2d_1a)
  conv2d_1a = keras.layers.Activation(relu6, name='ReLU_Conv2d_1a_%s'%str_modality)(conv2d_1a)

  conv2d_1b = keras.layers.SeparableConv2D(256, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_1b_%s'%str_modality)(conv2d_1a)
  conv2d_1b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_1b_%s'%str_modality)(conv2d_1b)
  conv2d_1b = keras.layers.Activation(relu6, name='ReLU_Conv2d_1b_%s'%str_modality)(conv2d_1b)

  conv2d_2a = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2a_%s'%str_modality)(conv2d_1b)
  conv2d_2a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2a_%s'%str_modality)(conv2d_2a)
  conv2d_2a = keras.layers.Activation(relu6, name='ReLU_Conv2d_2a_%s'%str_modality)(conv2d_2a)

  conv2d_2b = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2b_%s'%str_modality)(conv2d_2a)
  conv2d_2b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2b_%s'%str_modality)(conv2d_2b)
  conv2d_2b = keras.layers.Activation(relu6, name='ReLU_Conv2d_2b_%s'%str_modality)(conv2d_2b)

  conv2d_2c = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2c_%s'%str_modality)(conv2d_2b)
  conv2d_2c = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2c_%s'%str_modality)(conv2d_2c)
  conv2d_2c = keras.layers.Activation(relu6, name='ReLU_Conv2d_2c_%s'%str_modality)(conv2d_2c)

  conv2d_2d = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2d_%s'%str_modality)(conv2d_2c)
  conv2d_2d = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2d_%s'%str_modality)(conv2d_2d)
  conv2d_2d = keras.layers.Activation(relu6, name='ReLU_Conv2d_2d_%s'%str_modality)(conv2d_2d)

  conv2d_2e = keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_2e_%s'%str_modality)(conv2d_2d)
  conv2d_2e = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_2e_%s'%str_modality)(conv2d_2e)
  conv2d_2e = keras.layers.Activation(relu6, name='ReLU_Conv2d_2e_%s'%str_modality)(conv2d_2e)

  conv2d_3a = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3a_%s'%str_modality)(conv2d_2e)
  conv2d_3a = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3a_%s'%str_modality)(conv2d_3a)
  conv2d_3a = keras.layers.Activation(relu6, name='ReLU_Conv2d_3a_%s'%str_modality)(conv2d_3a)

  conv2d_3b = keras.layers.SeparableConv2D(1024, (3,3), strides=(2,2), padding='same',
                    depthwise_regularizer=l2(weight_decay), pointwise_regularizer=l2(weight_decay),
                    name='SeparableConv2D_3b_%s'%str_modality)(conv2d_3a)
  conv2d_3b = keras.layers.BatchNormalization(name='BatchNorm_Conv2d_3b_%s'%str_modality)(conv2d_3b)
  conv2d_3b = keras.layers.Activation(relu6, name='ReLU_Conv2d_3b_%s'%str_modality)(conv2d_3b)

  return conv2d_3b

def FusionRes3d_clstm_mobilenet(inputs_RGB, inputs_Flow, seq_len, weight_decay, str_modality):

  # Res3D Block
  res3d_featmap = FusionRes3d(inputs_RGB, inputs_Flow, weight_decay)

  # GatedConvLSTM2D Block
  clstm2d_1 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_1_%s'%str_modality)(res3d_featmap)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_2_%s'%str_modality)(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape_%s'%str_modality)(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay, str_modality)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape_%s'%str_modality)(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4),
                    padding='valid', name='Average_Pooling_%s'%str_modality)(features)

  return gpooling

def CatConvFusionRes3d_clstm_mobilenet(inputs_RGB, inputs_Flow, seq_len, weight_decay, str_modality):

  # Res3D Block
  res3d_featmap = CatConvFusionRes3d(inputs_RGB, inputs_Flow, weight_decay)

  # GatedConvLSTM2D Block
  clstm2d_1 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_1_%s'%str_modality)(res3d_featmap)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_2_%s'%str_modality)(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape_%s'%str_modality)(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay, str_modality)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape_%s'%str_modality)(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4),
                    padding='valid', name='Average_Pooling_%s'%str_modality)(features)

  return gpooling

def res3d_clstm_mobilenet(inputs, seq_len, weight_decay, str_modality):
  # Res3D Block
  res3d_featmap = res3d(inputs, weight_decay, str_modality)

  # GatedConvLSTM2D Block
  clstm2d_1 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_1_%s'%str_modality)(res3d_featmap)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_2_%s'%str_modality)(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape_%s'%str_modality)(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay, str_modality)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape_%s'%str_modality)(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4),
                    padding='valid', name='Average_Pooling_%s'%str_modality)(features)

  return gpooling

def clstm_mobilenet(inputs, seq_len, weight_decay, str_modality):

  # GatedConvLSTM2D Block
  clstm2d_1 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_1_%s'%str_modality)(inputs)

  clstm2d_2 = keras.layers.GatedConvLSTM2D(256, (3,3), strides=(1,1), padding='same',
                      kernel_initializer='he_normal', recurrent_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay),
                      return_sequences=True, name='gatedclstm2d_2_%s'%str_modality)(clstm2d_1)
  featmap_2d = keras.layers.Reshape((28,28,256), name='clstm_reshape_%s'%str_modality)(clstm2d_2)

  # MobileNet
  features = mobilenet(featmap_2d, weight_decay, str_modality)
  features = keras.layers.Reshape((seq_len/2,4,4,1024), name='feature_reshape_%s'%str_modality)(features)
  gpooling = keras.layers.AveragePooling3D(pool_size=(seq_len/2,4,4), strides=(seq_len/2,4,4),
                    padding='valid', name='Average_Pooling_%s'%str_modality)(features)

  return gpooling
