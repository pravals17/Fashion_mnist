import util
import model
import numpy as np
import tensorflow as tf
import math
import os
from sklearn.metrics import accuracy_score

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/work/cse479/shared/homework/01/', 'directory where MNIST is located')
flags.DEFINE_string('save_dir', '/lustre/work/cse479/praval395/homework1_mine', 'directory where model graph and weights are saved')
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_integer('max_epoch_num', 1000, '')
FLAGS = flags.FLAGS
batch_size = FLAGS.batch_size

def train(train_images, train_labels):
    ce_vals = []
    train_predictions = []
    for i in range(train_num_examples // batch_size):
        batch_xs = train_images[i*batch_size:(i+1)*batch_size, :]
        batch_ys = train_labels[i*batch_size:(i+1)*batch_size, :]       
        _, train_ce, train_predicted = session.run([train_op, net_loss_regularization, output], {x: batch_xs, y: batch_ys})
        ce_vals.append(train_ce)
        train_predictions.append(train_predicted)
        
    avg_train_ce = sum(ce_vals) / len(ce_vals)
    print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
    train_accuracy = accuracy_score(np.argmax(train_labels, axis=1), np.argmax(np.vstack(train_predictions), axis=1))
    print("TRAIN ACCURACY:",train_accuracy)

    
def validation(validation_images, validation_labels):
    ce_vals = []
    validation_predictions = []
    for i in range(validation_num_examples // batch_size):
        batch_xs = validation_images[i*batch_size:(i+1)*batch_size, :]
        batch_ys = validation_labels[i*batch_size:(i+1)*batch_size, :]
        val_ce, validation_predicted = session.run([net_loss_regularization, output], {x:batch_xs, y:batch_ys})
        ce_vals.append(val_ce)
        validation_predictions.append(validation_predicted)
        
    avg_val_ce = sum(ce_vals) / len(ce_vals)
    print('Validation CROSS ENTROPY: ' + str(avg_val_ce))
    validation_accuracy = accuracy_score(np.argmax(validation_labels, axis=1), np.argmax(np.vstack(validation_predictions), axis=1))
    print("Validation Accuracy:",validation_accuracy)
    
    return avg_val_ce, validation_accuracy
    
def test(test_images, test_labels):
    ce_vals = []
    conf_mxs = []
    test_predictions = []
    for i in range(test_num_examples // batch_size):
        batch_xs = test_images[i*batch_size:(i+1)*batch_size, :]
        batch_ys = test_labels[i*batch_size:(i+1)*batch_size, :]
        test_ce, conf_matrix, test_predicted = session.run([net_loss_regularization, confusion_matrix, output], {x:batch_xs, y:batch_ys})
        ce_vals.append(test_ce)
        conf_mxs.append(conf_matrix)
        test_predictions.append(test_predicted)
        
    avg_test_ce = sum(ce_vals) / len(ce_vals)
    print('Test Cross Entropy: ' + str(avg_test_ce))
    test_accuracy =  accuracy_score(np.argmax(test_labels, axis=1), np.argmax(np.vstack(test_predictions), axis=1))
    print('Test Accuracy:', test_accuracy)
    
    
full_train_images = np.load(FLAGS.data_dir + 'fmnist_train_data.npy')
full_train_labels = np.load(FLAGS.data_dir + 'fmnist_train_labels.npy')

#divide into train, validation and test 
full_train_labels = util.one_hot_encode(full_train_labels, 10)
full_train_data = np.concatenate((full_train_images, full_train_labels), axis = 1)

training_set, test_set = util.split_rows(full_train_data, 0.8)
training_set_new, validation_set = util.split_rows(training_set, 0.8)
training_data = np.hsplit(training_set_new, [784, 794]) 
train_images = training_data[0]
train_labels = training_data[1]
validation_data = np.hsplit(validation_set, [784,794])
validation_images = validation_data[0]
validation_labels = validation_data[1]
test_data = np.hsplit(test_set, [784,794])
test_images = test_data[0]
test_labels = test_data[1]

#size of train, validation and test
train_num_examples = train_images.shape[0]
validation_num_examples = validation_images.shape[0]
test_num_examples = test_images.shape[0]


#define cross entropy, placeholders, confusion_matrix and regularization loss funtions
tf.reset_default_graph()
x, y, output = model.create_model()

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output)
confusion_matrix = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
REG_COEFF = 0.001
if regularization_losses:
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)
    net_loss_regularization = tf.reduce_mean(total_loss)

global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

avg_val_loss = []   
patience_param = 0
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for epoch in range(FLAGS.max_epoch_num):
        print('Epoch: ' + str(epoch) )
        
        train(train_images, train_labels)
        
        validation_loss, validation_accuracy = validation(validation_images, validation_labels)
        avg_val_loss.append(validation_loss)
        
        test(test_images, test_labels)
        
        #early stopping
        if epoch > 1 and (avg_val_loss[-2] - avg_val_loss[-1]) > 0.0000001:
            epoch_saved = epoch
            patience_param = patience_param + 1
            if patience_param > 10:
                print('End.............................')
                break
                
    #saved_path = saver.save(session, os.path.join(FLAGS.save_dir, "homework1_mine"), global_step=global_step_tensor)
  
        
    
    
    