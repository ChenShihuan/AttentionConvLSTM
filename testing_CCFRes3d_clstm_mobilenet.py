import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from SEres3d_clstm_mobilenet_Fusion import CatConvFusionRes3d_clstm_mobilenet,SeBlock,CrossBlock,CatConvBlock,relu6
from callbacks import LearningRateScheduler 
from datagen import isoTrainImageGenerator, isoTestImageGenerator, isoFusionTrainImageGenerator, isoFusionTestImageGenerator
from datagen import jesterTrainImageGenerator, jesterTestImageGenerator
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model
from tensorflow.contrib.keras.python.keras.models import load_model
from tensorflow.contrib.keras.python.keras.utils.generic_utils import CustomObjectScope

##########################

#Cross Block

##########################

# Modality
RGB = 0
Depth = 1
Flow = 2
Fusion = 3

# Dataset
JESTER = 0
ISOGD = 1

cfg_modality = RGB
cfg_dataset = ISOGD

# if cfg_modality == RGB:
#     str_modality = 'rgb'
# elif cfg_modality == Depth:
#     str_modality = 'depth'
# elif cfg_modality == Flow:
#     str_modality = 'flow'
# elif cfg_modality==Fusion:
#   str_modality = 'fusion'

if cfg_dataset == JESTER:
    nb_epoch = 10
    init_epoch = 0
    seq_len = 16
    batch_size = 16
    num_classes = 27
    # dataset_name = 'jester_%s' % str_modality
    # training_datalist = './dataset_splits/Jester/train_%s_list.txt' % str_modality
    # testing_datalist = './dataset_splits/Jester/valid_%s_list.txt' % str_modality
elif cfg_dataset == ISOGD:
    nb_epoch = 40
    init_epoch = 0
    seq_len = 32
    batch_size = 2
    num_classes = 249
    dataset_name = 'isogr_CCF'
    RGB_testing_datalist = './dataset_splits/IsoGD/valid_rgb_list.txt'
    Flow_testing_datalist = './dataset_splits/IsoGD/valid_flow_list.txt'

weight_decay = 0.00005
model_prefix = './models_Rewrite_SEnet/Fusion/'


inputs_RGB = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))
inputs_Flow = keras.layers.Input(shape=(seq_len, 112, 112, 3), batch_shape=(batch_size, seq_len, 112, 112, 3))

feature = CatConvFusionRes3d_clstm_mobilenet(inputs_RGB, inputs_Flow, seq_len, weight_decay, 'fusion')

flatten = keras.layers.Flatten(name='Flatten_fusion')(feature)
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)
outputs = keras.layers.Activation('softmax', name='Output')(classes)

model = keras.models.Model(inputs=[inputs_RGB, inputs_Flow], outputs=outputs)
optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # model_Fusion  = keras.models.Model(inputs=inputs_RGB, outputs=outputs)
# print(model.summary())

# plot_model(model,to_file="./network_image/training_CatConvFusionRes3d_clstm_mobilenet_clstm_mobilenet.png",show_shapes=True)

# load pretrained model
RGB_pretrained_model = '%sisogr_CCF_weights.05-2.95.h5'%(model_prefix)
print 'Loading pretrained model from %s' % RGB_pretrained_model
model.load_weights(RGB_pretrained_model, by_name=False)


_,test_labels = data.load_iso_video_list(RGB_testing_datalist)
test_steps = len(test_labels)/batch_size

if cfg_dataset == JESTER:
    # model.fit_generator(jesterTrainImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality),
    #                     steps_per_epoch=train_steps,
    #                     epochs=nb_epoch,
    #                     verbose=1,
    #                     callbacks=callbacks,
    #                     validation_data=jesterTestImageGenerator(
    #                         testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
    #                     validation_steps=test_steps,
    #                     initial_epoch=init_epoch,
    #                     )
    print 'JESTER still need FLOW dataset to use it'
elif cfg_dataset == ISOGD:
    print model.evaluate_generator(isoFusionTestImageGenerator(RGB_testing_datalist, Flow_testing_datalist, batch_size, seq_len, num_classes),
                        steps=test_steps,
                        )

