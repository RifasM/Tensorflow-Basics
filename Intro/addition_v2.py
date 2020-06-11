import os
import tensorflow as tf

# Turn off warning messages in program output
os.environ['TF_CAP_MIN_LOG_LEVEL'] = '2'

# Turn off eager execution
tf.compat.v1.disable_eager_execution()

# Define Computational Graph
X = tf.compat.v1.placeholder(tf.float32, name="X")
Y = tf.compat.v1.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="Addition")

with tf.compat.v1.Session() as session:

    results = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})

    print(results)
