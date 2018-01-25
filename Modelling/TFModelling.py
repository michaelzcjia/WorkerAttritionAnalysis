from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import tensorflow as tf
sys.path.append("../DataProcessing")
sys.path.append("../Data")
sys.path.append('../Models')
sys.path.insert(0, '../DataProcessing/DataRetriever.py')
import DataCleansing

FLAGS = None

def main():
  dr = DataCleansing.DataFrame()


  normData = dr.normalize()
  inputX, inputY = dr.getNeuralNetworkInputs(normData)

  # Import data
  learning_rate = 0.0001
  training_epochs = 1000
  display_step = 50
  n_samples = 10000

  # Create the model
  x = tf.placeholder(tf.float32, [None, 18]) #Create a tensor with the shape [None, 18], will later become the feature vectors
  w1 = tf.Variable(tf.zeros([18, 50])) #Weight tensor initialized as zeros to create the first layer
  b = tf.Variable(tf.random_uniform([50,])) #Bias has a shape of 100 so we can add them to the output of matrix multiplication
  layer1  = tf.matmul(x, w1) + b #Matmul does matrix multiplication. The first output layer values
  w2 = tf.Variable(tf.random_uniform([50, 2])) #Weight tensor for the second layer, outputs two values corresponding to classifications
  b2 = tf.Variable(tf.random_uniform([2,])) #Bias is a vector of length 2 to add to the output
  output = tf.matmul(layer1, w2) + b2 #Prediction layer to be rounded to get classification
  correctLabels = tf.placeholder(tf.float32, [None, 2])
  # Cost function: Mean squared error, this needs to be
  cost = tf.reduce_sum(tf.pow(correctLabels - output, 2)) / (2 * n_samples)
  # Gradient descent
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Initialize variabls and tensorflow session
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for i in range(training_epochs):
      sess.run(optimizer, feed_dict={x: inputX, correctLabels: inputY}) # Take a gradient descent step using our inputs and labels

      # Display logs per epoch step
      if (i) % display_step == 0:
          cc = sess.run(cost, feed_dict={x: inputX, correctLabels:inputY})
          print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)) #, \"W=", sess.run(W), "b=", sess.run(b)

  correctPrediction = tf.equal(tf.argmax(output, 1), tf.argmax(correctLabels, 1))
  accuracy = tf.reduce_mean(tf.cast(correctPrediction, 'float'))
  print(correctLabels)
  print(output)
  print('Accuracy:', accuracy.eval({x: inputX, correctLabels: inputY}, session=sess))

  # print ("Optimization Finished!")
  # training_cost = sess.run(cost, feed_dict={x: inputX, correctLabels: inputY})
  # print ("Training cost=", training_cost, "W=", sess.run(w1), "b=", sess.run(b), '\n')


if __name__ == "__main__":
  main()