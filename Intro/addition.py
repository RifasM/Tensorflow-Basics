import os
import tensorflow as tf

# Turn off warning messages in program output
os.environ['TF_CAP_MIN_LOG_LEVEL'] = '2'

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="Addition")

with tf.Session() as session:

    results = session.run(addition, feed_dict={X: [1], Y: [4]})

    print(results)

# tf.compat.v1.disable_eager_execution()
# tf_upgrade_v2 --infile=addition.py --outfile=addition_v2.py
