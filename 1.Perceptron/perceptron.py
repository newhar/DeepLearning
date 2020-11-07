import tensorflow as tf

x_data = [[0.1, 2.3, 3.5]]

X = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# step
step_hypothesis = tf.maximum(0.0, tf.sign(tf.matmul(X, W) + b))
# Logistic(sigmoid)
sig_hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# relu
relu_hypothesis = tf.nn.relu(tf.matmul(X, W) + b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    prediction = sess.run(step_hypothesis, feed_dict={X: x_data})
    print('step function')
    print(prediction)

    prediction = sess.run(sig_hypothesis, feed_dict={X: x_data})
    print('sigmoid function')
    print(prediction)

    prediction = sess.run(relu_hypothesis, feed_dict={X: x_data})
    print('relu function')
    print(prediction)