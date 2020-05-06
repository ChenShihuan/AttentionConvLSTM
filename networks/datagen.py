import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
import inputs as data
import threading

## Iteration
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batch_size]
    else:
      excerpt = slice(start_idx, start_idx + batch_size)
    yield inputs[excerpt], targets[excerpt]

## Threading
def threading_data(data=None, fn=None, **kwargs):
  # define function for threading
  def apply_fn(results, i, data, kwargs):
    results[i] = fn(data, **kwargs)

  ## start multi-threaded reading.
  results = [None] * len(data) ## preallocate result list
  threads = []
  for i in range(len(data)):
    t = threading.Thread(
                    name='threading_and_return',
                    target=apply_fn,
                    args=(results, i, data[i], kwargs)
                    )
    t.start()
    threads.append(t)

  ## <Milo> wait for all threads to complete
  for t in threads:
    t.join()

  return np.asarray(results)

## isoTrainImageGenerator
def isoTrainImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
  y_train = np.asarray(y_train, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train,
                                            batch_size, shuffle=True):
      # Read data for each batch
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)



## isoTestImageGenerator
def isoTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test,
                                            batch_size, shuffle=False):
      # Read data for each batch
      image_path = []
      image_fcnt = []
      image_olen = []
      image_start = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,image_start,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_rgb_data)
      elif modality==1: #Depth
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_depth_data)
      elif modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_iso_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

## jesterTrainImageGenerator
def jesterTrainImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_train,y_train = data.load_iso_video_list(filepath)
  X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
  y_train = np.asarray(y_train, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_tridx, y_train,
                                            batch_size, shuffle=True):
      # Read data for each batch
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_train[key_str]['videopath'])
        image_fcnt.append(X_train[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(True) # Training
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)

## jesterTestImageGenerator
def jesterTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality):
  X_test,y_test = data.load_iso_video_list(filepath)
  X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
  y_test  = np.asarray(y_test, dtype=np.int32)
  while 1:
    for X_indices, y_label_t in minibatches(X_teidx, y_test,
                                            batch_size, shuffle=False):
      # Read data for each batch
      image_path = []
      image_fcnt = []
      image_olen = []
      is_training = []
      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path.append(X_test[key_str]['videopath'])
        image_fcnt.append(X_test[key_str]['framecnt'])
        image_olen.append(seq_len)
        is_training.append(False) # Testing
      image_info = zip(image_path,image_fcnt,image_olen,is_training)
      if modality==0: #RGB
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_rgb_data)
      if modality==2: #Flow
        X_data_t = threading_data([_ for _ in image_info], data.prepare_jester_flow_data)
      y_hot_label_t = keras.utils.to_categorical(y_label_t, num_classes=num_classes)
      yield (X_data_t, y_hot_label_t)



# (x1,y1) = jesterTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality)

# (x2,y2) = jesterTestImageGenerator(filepath, batch_size, seq_len, num_classes, modality)

# [x1,]
# ## isoFusionTrainImageGenerator
def isoFusionTrainImageGenerator(filepath_1, filepath_2, batch_size, seq_len, num_classes):

  # filepath_1:RGB, filepath_2:Flow
  X_train_1,y_train_1 = data.load_iso_video_list(filepath_1)
  X_train_2,y_train_2 = data.load_iso_video_list(filepath_2)
  X_tridx_1 = np.asarray(np.arange(0, len(y_train_1)), dtype=np.int32)
  X_tridx_2 = np.asarray(np.arange(0, len(y_train_2)), dtype=np.int32)
  y_train_1 = np.asarray(y_train_1, dtype=np.int32)
  y_train_2 = np.asarray(y_train_2, dtype=np.int32)

  # RGB
  while 1:
    for X_indices, y_label_t_1 in minibatches(X_tridx_1, y_train_1, batch_size, shuffle=True):
                                      # minibatches(X_tridx_2, y_train_2, batch_size, shuffle=True)):

      # Read data for each batch
      image_path_1 = []
      image_fcnt_1 = []
      image_path_2 = []
      image_fcnt_2 = []
      image_olen = []
      image_start = []
      is_training = []

      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path_1.append(X_train_1[key_str]['videopath'])
        image_fcnt_1.append(X_train_1[key_str]['framecnt'])
        image_path_2.append(X_train_2[key_str]['videopath'])
        image_fcnt_2.append(X_train_2[key_str]['framecnt'])
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(True) # Training
      image_info_1 = zip(image_path_1,image_fcnt_1,image_olen,image_start,is_training)
      image_info_2 = zip(image_path_2,image_fcnt_2,image_olen,image_start,is_training)

      X_data_t_1 = threading_data([_ for _ in image_info_1], data.prepare_iso_rgb_data)
      X_data_t_2 = threading_data([_ for _ in image_info_2], data.prepare_iso_flow_data)
      y_hot_label_t_1 = keras.utils.to_categorical(y_label_t_1, num_classes=num_classes)
      # import keras.backend as K
      # X_data_t = keras.layers.Concatenate(axis=-1)([X_data_t_1, X_data_t_2])
      # print(X_data_t_1.shape)
      # print(X_data_t_2.shape)
      # X_data_t = tf.concat([X_data_t_1, X_data_t_2], axis=-1)
     
      # print('*************')
      yield ([X_data_t_1, X_data_t_2],  y_hot_label_t_1)


## isoFusionTestImageGenerator
def isoFusionTestImageGenerator(filepath_1, filepath_2, batch_size, seq_len, num_classes):

  # filepath_1:RGB, filepath_2:Flow
  X_train_1,y_train_1 = data.load_iso_video_list(filepath_1)
  X_train_2,y_train_2 = data.load_iso_video_list(filepath_2)
  X_tridx_1 = np.asarray(np.arange(0, len(y_train_1)), dtype=np.int32)
  X_tridx_2 = np.asarray(np.arange(0, len(y_train_2)), dtype=np.int32)
  y_train_1 = np.asarray(y_train_1, dtype=np.int32)
  y_train_2 = np.asarray(y_train_2, dtype=np.int32)

  # RGB
  while 1:
    for X_indices, y_label_t_1 in minibatches(X_tridx_1, y_train_1, batch_size, shuffle=True):
                                      # minibatches(X_tridx_2, y_train_2, batch_size, shuffle=True)):

      # Read data for each batch
      image_path_1 = []
      image_fcnt_1 = []
      image_path_2 = []
      image_fcnt_2 = []
      image_olen = []
      image_start = []
      is_training = []

      for data_a in range(batch_size):
        X_index_a = X_indices[data_a]
        key_str = '%06d' % X_index_a
        image_path_1.append(X_train_1[key_str]['videopath'])
        image_fcnt_1.append(X_train_1[key_str]['framecnt'])
        image_path_2.append(X_train_2[key_str]['videopath'])
        image_fcnt_2.append(X_train_2[key_str]['framecnt'])
        image_olen.append(seq_len)
        image_start.append(1)
        is_training.append(False) # Testing
      image_info_1 = zip(image_path_1,image_fcnt_1,image_olen,image_start,is_training)
      image_info_2 = zip(image_path_2,image_fcnt_2,image_olen,image_start,is_training)

      X_data_t_1 = threading_data([_ for _ in image_info_1], data.prepare_iso_rgb_data)
      X_data_t_2 = threading_data([_ for _ in image_info_2], data.prepare_iso_flow_data)
      y_hot_label_t_1 = keras.utils.to_categorical(y_label_t_1, num_classes=num_classes)
      # import keras.backend as K
      # X_data_t = keras.layers.Concatenate(axis=-1)([X_data_t_1, X_data_t_2])
      # print(X_data_t_1.shape)
      # print(X_data_t_2.shape)
      # X_data_t = tf.concat([X_data_t_1, X_data_t_2], axis=-1)
     
      # print('*************')
      yield ([X_data_t_1, X_data_t_2],  y_hot_label_t_1)