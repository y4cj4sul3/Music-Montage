import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

batch_size = 32

n_seg = 10 # num. of segments
m_cate = 25 # num. of category of clips
seg_len = 100 # segment length

steps = 1000
temperature = 0.5 # gumble softmax
sess = tf.InteractiveSession()

# load data 
X = np.random.randn(batch_size, n_seg, seg_len) * 5 + 50 # n segments
SB = np.random.randn(batch_size, m_cate, seg_len) * 5 + 50 # m categories of clips


# model
# ----------------------------------------
x_input = tf.placeholder(tf.float32, shape=(None, n_seg, seg_len))
sb_input = tf.placeholder(tf.float32, shape=(None, m_cate, seg_len))

# rnn layer
lstm_cell = tf.nn.rnn_cell.LSTMCell(seg_len)
initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
output, (c_t, h_t) = tf.nn.dynamic_rnn(lstm_cell, x_input,
                                  initial_state=initial_state,
                                  dtype=tf.float32)
state = tf.concat([c_t, h_t], axis=-1)

# z dist
z = tf.contrib.layers.fully_connected(state, m_cate * n_seg)
z = tf.reshape(z, [batch_size, n_seg, m_cate])
z = tf.nn.softmax(z)

# synthesizer
x_output = tf.matmul(z, sb_input)

# loss and optimizer
loss = tf.losses.mean_squared_error(x_input, x_output)
opt = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = opt.minimize(loss)


# main
# ----------------------------------------
sess.run(tf.initializers.global_variables())
print(X.shape)
print(SB.shape)

for i in range(steps):
    x_, _ = sess.run([x_output, train_op], feed_dict={
        x_input: X, sb_input: SB
    })
