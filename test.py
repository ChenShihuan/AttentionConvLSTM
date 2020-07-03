import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
from keras import backend as K

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
                                      shape=[self.batch_size,self.channel],
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

        xAA = keras.layers.Multiply()([self.alphaAA, inputs])
        xAB = keras.layers.Multiply()([self.alphaAB, xAA])

        # xa_estimated = keras.layers.Add(name='xa_estimated')([xAA, xAB])
        
        return xAB

x = tf.ones((3, 3, 3))
linear_layer = keras.layers.Conv1D(filters = 1, kernel_size = 1, strides=1, padding='same',
                              dilation_rate=1, kernel_initializer='he_normal',
                              kernel_regularizer=l2(0.00005), activity_regularizer=None,
                              kernel_constraint=None, use_bias=False)
y = linear_layer(x)
print(y)
print('weights:', len(linear_layer.weights))
print('trainable weights:', len(linear_layer.trainable_weights))

class MLPBlock(keras.layers.Layer):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.CrossStitch_1 = CrossStitchBlock(batch_size = 3, channel = 32)
        self.CrossStitch_2 = CrossStitchBlock(batch_size = 3, channel = 32)
        self.CrossStitch = CrossStitchBlock(batch_size = 3, channel = 32)

    def build(self, input_shape):

        super(MLPBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.CrossStitch_1(inputs)
        x = self.CrossStitch_2(x)
        return self.CrossStitch(x)

class SeBlock(keras.layers.Layer):
    def __init__(self, channel, reduction_ratio=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.channel = channel
        self.GlobalAveragePooling3D = keras.layers.GlobalAveragePooling3D(name='SE_Global_Average_Pooling')
        self.Dense_1 = keras.layers.Dense(self.channel / self.reduction_ratio, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1a')
        self.Activation = keras.layers.Activation('relu', name='SE_ReLU_1')
        self.Dense_2 = keras.layers.Dense(self.channel, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(self.reduction_ratio), name='Fully_connected_1b')
        self.sigmoid = keras.backend.sigmoid
        self.Reshape = keras.layers.Reshape([1,1,1,self.channel], name='SE_Reshape')
        self.Multiply = keras.layers.Multiply()

    def build(self,input_shape):
      # input_shape
      super(SeBlock, self).build(input_shape)

    def call(self, inputs):
        x = keras.backend.mean(inputs, axis=[1, 2, 3])
        x = keras.backend.
        x = keras.backend.relu(x, alpha=0.0, max_value=None)
        x = keras.backend.
        x = keras.backend.

        x = self.Dense_1(x)
        x = self.Activation(x)
        x = self.Dense_2(x)
        x = self.sigmoid(x)
        x = self.Reshape(x)

        return self.Multiply([inputs,x])

mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 32)))  # The first call to the `mlp` will create the weights
print(y)
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))