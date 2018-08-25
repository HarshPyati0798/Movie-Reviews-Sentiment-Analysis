import tensorflow as tf
import numpy as np

from sentiment_analysis import create_feature_sets_and_labels
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2 # Either a positive or negative sentiment
batch_size = 100

X = tf.placeholder(tf.float32,[None,len(train_x[0])])
y = tf.placeholder(tf.float32)

def neural_network(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases' :tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)
    
    layer_2 = tf.add(tf.matmul(layer_1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)
    
    layer_3 = tf.add(tf.matmul(layer_2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)
    
    output_layer = tf.matmul(layer_3,output_layer['weights']) +output_layer['biases']
    
    return output_layer

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels =y))
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    nm_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(nm_epochs):
            epoch_loss = 0
            i = 0
            while i<len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch:',epoch+1,' completed out of -',nm_epochs,' loss:',epoch_loss)
            
        
        # RUNNING THE MODEL ON THE TEST SET
        
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        
        print('ACCURACY :',accuracy.eval({X:test_x,y:test_y}))

train_neural_network(X)
