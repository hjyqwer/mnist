from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.nn.softmax(tf.matmul(x,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

training_epochs = 25
batch_size = 100
display_step = 1

saver = tf.train.Saver()
saver_dir = "log/"

tf.summary.histogram('pred',pred)
tf.summary.scalar('accuracy',accuracy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/mnist_summaries',sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost+=c/total_batch

        saver.save(sess,saver_dir+"mnistmodel.cpkt",global_step=epoch)
        if(epoch+1)%display_step == 0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost))

        summary_str = sess.run(merged_summary_op, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        summary_writer.add_summary(summary_str, epoch)
    print("Finished!")

    print("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))