#from _future_ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as t_f

# Import new dataset not used in class : MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
#saving the data
m = input_data.read_data_sets("", one_hot=True)

# Parameters Defined!
l_r = 0.01  # learning rate for the dataset
t_e = 50    # number of training epochs used
b_s = 1000    # batch size for training data
d_s = 10  # step size for displaying epochs

# Tensor Flow Graph Input
# mnist data image of shape 28*28=784 in the database
x = t_f.placeholder(t_f.float32, [None, 784])
# Variable needs to recognize 0-9 digits i.e, 10 classes
y = t_f.placeholder(t_f.float32, [None, 10])

# Setting Model Weights for the Dataset
Wei = t_f.Variable(t_f.zeros([784, 10]))
bi = t_f.Variable(t_f.zeros([10]))

# Prediction Model Construction
pred = t_f.nn.softmax(t_f.matmul(x, Wei) + bi)

# Cross entropy is used in reducing the error
cst = t_f.reduce_mean(-t_f.reduce_sum(y*t_f.log(pred), reduction_indices=1))

# Gradient Descent
optm = t_f.train.GradientDescentOptimizer(l_r).minimize(cst)

# Initialize the variables (i.e. assign their default value)
st = t_f.global_variables_initializer()

# Training session begins here
with t_f.Session() as sess:

    # Run the initializer
    sess.run(st)
    writer = t_f.summary.FileWriter('./graphs/logistic_reg', sess.graph)

   # Training cycle
    for epoch in range(t_e):
        ag_ct = 0.
        t_b = int(m.train.num_examples/b_s) #total batch taken

        # Loop over all batches
        for i in range(t_b):
            batch_xs, batch_ys = m.train.next_batch(b_s)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optm, cst], feed_dict={x: batch_xs, y: batch_ys})

            # Average cost calculation
            ag_ct += c / t_b

        # Display logs per epoch step
        if (epoch+1) % d_s == 0:

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(ag_ct))

    print("Complete Optimization Done!")

 # Designed Test Model
    # correction prediction determined by taking maximum of the dataset
    c_p = t_f.equal(t_f.argmax(pred, 1), t_f.argmax(y, 1))
 #Accuracy Calculation for the developed epochs
    acc = t_f.reduce_mean(t_f.cast(c_p, t_f.float32))
    print("Accuracy of the MNIST Dataset:", acc.eval({x: m.test.images, y: m.test.labels}))