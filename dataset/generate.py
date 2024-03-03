import tensorflow as tf
import numpy as np

# 加载之前训练好的模型
model = tf.keras.models.load_model('music_generation_model.h5')

# 生成音乐
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]

# 输出生成的音乐
