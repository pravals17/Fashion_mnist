import tensorflow as tf

def create_model():
    #tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name = "input_placeholder")
    x = x/255.0
    y = tf.placeholder(tf.float32, [None, 10], name = 'label')
    
    with tf.name_scope('linear_model') as scope:
        dropout_1 = tf.layers.dropout(x, rate = 0.2, name='dropout_1')
        hidden_1 = tf.layers.dense(x, 400, activation = tf.nn.relu,
                                   kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                   bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                   name = 'hidden_layer_1')
        dropout_2 = tf.layers.dropout(hidden_1, rate = 0.2, name = 'dropout_2')
        hidden_2 = tf.layers.dense(dropout_2, 400, activation = tf.nn.relu,
                                    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                    bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.01),
                                    name = 'hidden_layer_2')
        dropout_3 = tf.layers.dropout(hidden_2, rate = 0.2, name = 'dropout_3')
        hidden_3 = tf.layers.dense(dropout_3, 300, activation = tf.nn.relu,
                                   kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                   bias_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01),
                                   name = 'dropout_3')
        dropout_4 = tf.layers.dropout(hidden_3, rate = 0.1, name = 'dropout_4')
        output = tf.layers.dense(dropout_4, 10, name = 'output')
    tf.identity(output, name='output')
    return x, y, output
        