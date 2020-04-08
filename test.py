from tensorflow.contrib.keras.python.keras.models import load_model
model = load_model('/home/chenshihuan/AttentionConvLSTM/models/jester_rgb_gatedclstm_weights.h5')
print(model.summary())#输出网络的结构图
