'''
Speech Emotion Recognition using Deep Learning
Uses IEMOCAP Dataset
Copyright @2018 Aman Agarwal, Aditya Mishra
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

# setting seed to get reproducable results
tf.set_random_seed(1234)
np.random.seed(1234)

# Reading Data file
df = pd.read_pickle("speech_emotion_data.pkl", compression='gzip')
df.reset_index(inplace=True)
df.drop('index', inplace=True, axis=1)
        
# setting sequence lenghts
for index, row in df.iterrows():
    df.set_value(index, 'lengths', row['Features'].shape[0])
    
# coverting labels to numeric values
le = LabelEncoder()
le.fit(df['Label'].values)
le.classes_

# preparing data as input
df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
print('training samples:', df_train.shape[0])
print('test samples:', df_test.shape[0])

# Bucket Iterator
class BucketedDataIterator():
    def __init__(self, df, num_buckets = 7):
        df = df.sort_values('lengths').reset_index(drop=True)
        self.size = len(df) / num_buckets
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.loc[bucket*self.size: (bucket+1)*self.size - 1])
        self.num_buckets = num_buckets

        # cursor[i] will be the cursor for the ith bucket
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()

        self.epochs = 0

    def shuffle(self):
        #sorts dataframe by sequence length, but keeps it random within the same length
        for i in range(self.num_buckets):
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0

    def next_batch(self, n):
        if np.any(self.cursor+n+1 > self.size):
            self.epochs += 1
            self.shuffle()

        i = np.random.randint(0,self.num_buckets)

        res = self.dfs[i].loc[self.cursor[i]:self.cursor[i]+n-1]
        self.cursor[i] += n

        # Pad sequences with 0s so they are all the same length
        maxlen = int(max(res['lengths']))
        
        x = np.zeros([n, maxlen, 32], dtype=np.float)
        for i, x_i in enumerate(x):
            x_i[:res['lengths'].values[i]] = res['Features'].values[i]

        return x, le.transform(res['Label'].values), res['lengths'].values

# some hyperparameters
init_epoch = 0          # starting epochs when loading weights
hm_epochs = 500         # maximum epochs
n_classes = 6           # number of classes
batch_size = 16         # batch size
learning_rate = 0.001   # learning rate
num_mfcc_features = 32  # number of features
rnn_size = 64           # number of cells in LSTM layers
dropout = 0.8           # fraction of neurons to keep while training
epsilon = 1e-3          # for batchnorm
decay = 0.999           # decay for batchnorm
attn_length = 64        # inputs to consider for attention
num_layers = 3          # number of LSTM layers

# defining placeholders
x = tf.placeholder(tf.float32,[batch_size, None, num_mfcc_features])
y = tf.placeholder(tf.int32, [batch_size])
seq_length = tf.placeholder(tf.int32, [batch_size])
keep_prob = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool)       # checking for training/testing phase

scale = tf.Variable(tf.ones([128]))
beta = tf.Variable(tf.zeros([128]))
pop_mean = tf.Variable(tf.zeros([128]), trainable=False)
pop_var = tf.Variable(tf.ones([128]), trainable=False)

layer = {'weight':tf.Variable(tf.random_normal([rnn_size*2, n_classes])),
        'bias':tf.Variable(tf.random_normal([n_classes]))}

# tr_acc, te_acc, tr_loss, te_loss = [], [], [], []
# tr_acc = np.load('tr_acc.npy')
# tr_loss = np.load('tr_loss.npy')
# te_acc = np.load('te_acc.npy')
# te_loss = np.load('te_loss.npy')

# model graph
def feed_forward(inputs, seq_length):
    
    # multiple RNN layers 
    stacked_rnn_fw, stacked_rnn_bw = [], []
    for _ in range(num_layers-1):
        stacked_rnn_fw.append(rnn.BasicLSTMCell(rnn_size, state_is_tuple=True))
        stacked_rnn_bw.append(rnn.BasicLSTMCell(rnn_size, state_is_tuple=True))
    # adding attention at last LSTM layer
    stacked_rnn_fw.append(tf.contrib.rnn.AttentionCellWrapper(rnn.BasicLSTMCell(rnn_size, state_is_tuple=True), attn_length=attn_length, state_is_tuple=True))
    stacked_rnn_bw.append(tf.contrib.rnn.AttentionCellWrapper(rnn.BasicLSTMCell(rnn_size, state_is_tuple=True), attn_length=attn_length, state_is_tuple=True))
    # stacking LSTM layers
    cell_fw = rnn.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
    cell_bw = rnn.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
    # applying bidirectional wrapper
    rnn_outputs, final_state_fw, final_state_bw = stack_bidirectional_dynamic_rnn([cell_fw], [cell_bw], inputs, dtype=tf.float32, sequence_length=seq_length)
    print('rnn_outputs shape:',rnn_outputs.shape)
  
    # applying batch normalization
    if phase == True:      # while training
        batch_mean, batch_var = tf.nn.moments(rnn_outputs,[0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            rnn_outputs = tf.nn.batch_normalization(rnn_outputs, batch_mean, batch_var, beta, scale, epsilon)
    else:                 # while validation
        rnn_outputs = tf.nn.batch_normalization(rnn_outputs, pop_mean, pop_var, beta, scale, epsilon)

    rnn_outputs = tf.nn.relu(rnn_outputs)
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    
    # stacking the outputs of all time frames
    last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seq_length-1], axis=1))
    
    # output layer
    output = tf.add(tf.matmul(last_rnn_output, layer['weight']), layer['bias'])
    print('Output shape', output.get_shape())
    
    return output
    
# computes the average of gradients across towers
def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # grad_and_vars: ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Variables are redundant because they are shared across towers.
        # We will return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

# computing losses for every tower(GPU)
def losses(scope, inputs, labels, seq_length):

    with tf.device('/device:CPU:0'):
        labels = tf.one_hot(labels,6)

    # feed forward
    logits = feed_forward(inputs, seq_length)

    # calculating accuracy
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(preds,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss, accuracy

# training model
def train_model():

    with tf.Graph().as_default(), tf.device('/device:CPU:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        with tf.variable_scope("Scope_adam", reuse=True):
            optimizer = tf.train.AdamOptimizer(learning_rate)

        # mini_batch = tr.next_batch(batch_size)
        # dataset = tf.data.Dataset.from_tensor_slices((mini_batch[0], mini_batch[1], mini_batch[2])).repeat().batch(batch_size)
        # iter1 = dataset.make_one_shot_iterator()
        
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            for i in range(2): 
                with tf.device('/device:GPU:%d' % i):
                    with tf.name_scope('%s_%d' % ("Tower", i)) as scope:

                        # el = iter1.get_next()

                        # print('x shape:',el[0].get_shape())
                        # print('y shape:',el[1].get_shape())
                        # print('seq_len shape:',el[2].get_shape())
                        loss, acc = losses(scope, x, y, seq_length)
                        loss = tf.cast(loss, tf.float32)
                        
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        grads = optimizer.compute_gradients(loss)
                        # setting gradient sequences
                        if i==1:
                            grads[0:7] = grads[-7:]
                            grads = grads[:-7]

                        tower_grads.append(grads)
             
        # computing average of gradients           
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # appling EMA to variables
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
    
        # initializing the variables
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        tr = BucketedDataIterator(df_train, 1)
        te = BucketedDataIterator(df_test, 1)

        # for epoch in range(hm_epochs):
        tr_acc, te_acc, tr_loss, te_loss = [], [], [], []

        step, accuracy, loss = 0, 0, 0
        current_epoch = 0
        ep = False

        saver = tf.train.Saver()

        while current_epoch < hm_epochs:
            # for timing the epoch
            if ep==False:
                start_time = time.time()
                ep = True
            step += 1

            mini_batch = tr.next_batch(batch_size)
            dataset = tf.data.Dataset.from_tensor_slices((mini_batch[0], mini_batch[1], mini_batch[2])).repeat().batch(batch_size)
            iter_data = dataset.make_one_shot_iterator().get_next()

            _, loss_, accuracy_ = sess.run([train_op, loss, acc], feed_dict={'x': iter_data[0], 'y': iter_data[1],
                                                                  'seq_length': iter_data[2], 'phase': True, 'dropout': dropout})
            accuracy += accuracy_
            loss += loss_ 
            
            if tr.epochs > current_epoch:
                ep = False
                current_epoch += 1
                tr_acc = np.append(tr_acc, accuracy/step)
                tr_loss = np.append(tr_loss, loss/step)
                step, accuracy, loss = 0, 0, 0

                # eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    mini_batch = te.next_batch(batch_size)
                    dataset = tf.data.Dataset.from_tensor_slices((mini_batch[0], mini_batch[1], mini_batch[2])).repeat().batch(batch_size)
                    iter_data = dataset.make_one_shot_iterator().get_next()

                    loss_, accuracy_ = sess.run([loss, acc], feed_dict={'x': iter_data[0], 'y': iter_data[1],
                                                             'seq_length': iter_data[2], 'phase': False, 'dropout': 1})

                    accuracy += accuracy_
                    loss += loss_

                te_acc = np.append(te_acc, accuracy/step)
                te_loss = np.append(te_loss, loss/step)
                step, accuracy, loss = 0, 0, 0
                
                if current_epoch%20==0:
                    saver.save(sess, 'C:\\Users\\Administrator\\Desktop\\aman_speech\\temp_64_att_3_lr\\weights_epoch_{0}.ckpt'.format(current_epoch+init_epoch))
                    np.save('C:\\Users\\Administrator\\Desktop\\aman_speech\\temp_64_att_3_lr\\train_accuracy.npy', tr_acc)
                    np.save('C:\\Users\\Administrator\\Desktop\\aman_speech\\temp_64_att_3_lr\\train_loss.npy', tr_loss)
                    np.save('C:\\Users\\Administrator\\Desktop\\aman_speech\\temp_64_att_3_lr\\test_accuracy.npy', te_acc)
                    np.save('C:\\Users\\Administrator\\Desktop\\aman_speech\\temp_64_att_3_lr\\test_loss.npy', te_loss)
                    print('Weights saved.')
               
                end_time = time.time()
                print("Epoch : %d/%d - time: %.2fs\nloss: %0.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f\n" % (current_epoch+init_epoch, hm_epochs+init_epoch, end_time-start_time, tr_loss[-1], tr_acc[-1], te_loss[-1], te_acc[-1]))

train_model()
