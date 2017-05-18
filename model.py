# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from rnn_cell import *
from constants import *

def create_model(index, model_type):
  device = "/gpu:0"

  if model_type == "rnn_action":
    network = SomaticActionRecurrentNetwork(ACTION_SIZE, index, device)
  elif model_type == "rnn":
    network = SomaticRecurrentNetwork(ACTION_SIZE, index, device)
  elif model_type == "plain_cnn":
    network = SomaticCNNNetwork(ACTION_SIZE, index, device)
  elif model_type == "rnn_cnn":
    network = SomaticRecurrentCNNNetwork(ACTION_SIZE, index, device)
  else:
    network = SomaticSimpleNetwork(ACTION_SIZE, index, device)
  
  return network

def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer

class SomaticNetwork(object):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0"):
    self._action_size = action_size
    self._thread_index = thread_index
    self._device = device

  def prepare_loss(self, entropy_beta):
    with tf.device(self._device):
      # taken action (input for policy)
      self.a = tf.placeholder("float", [None, self._action_size])
    
      # temporary difference (R-V) (input for policy)
      self.td = tf.placeholder("float", [None])

      # avoid NaN with clipping when value in pi becomes zero
      log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
      
      # policy entropy
      entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
      
      # policy loss (output)
      policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), reduction_indices=1 ) * self.td + entropy * entropy_beta )

      # R (input for value)
      self.r = tf.placeholder("float", [None])
      
      # value loss (output)
      # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

      # gradienet of policy and value are summed up
      self.total_loss = policy_loss + value_loss

  def run_policy_and_value(self, sess, s_t, last_action):
    raise NotImplementedError()

  def run_value(self, sess, s_t, last_action):
    raise NotImplementedError()

  def get_vars(self):
    raise NotImplementedError()

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "SomaticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  def _fc_variable(self, weight_shape):
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
    return weight, bias

  def _conv_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels  = weight_shape[2]
    output_channels = weight_shape[3]
    bias_shape = [output_channels]
    
    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")
  
class SomaticSimpleNetwork(SomaticNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    SomaticNetwork.__init__(self, action_size, thread_index, device)
    scope_name = "net_" + str(self._thread_index)
    
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_fc1, self.b_fc1 = self._fc_variable([100, 256]) #
      
      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])
      s_flat = tf.reshape(self.s, [-1, 100])

      h_fc1 = tf.nn.relu(tf.matmul(s_flat, self.W_fc1) + self.b_fc1)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )
      
  def run_policy_and_value(self, sess, s_t, last_action):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])
                                           
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_value(self, sess, s_t, last_action):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]


class SomaticRecurrentNetwork(SomaticNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    SomaticNetwork.__init__(self, action_size, thread_index, device)
    scope_name = "net_" + str(self._thread_index)
    
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_fc1, self.b_fc1 = self._fc_variable([100, 256]) #

      self.W_fc2, self.b_fc2 = self._fc_variable([256, 256]) # 
      
      # weight for policy output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc4, self.b_fc4 = self._fc_variable([256, 1])

      # rnn cell
      self.cell = SomaticRNNCell(256)

      # place holder for RNN unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])
      s_flat = tf.reshape(self.s, [-1, 100])

      self.initial_rnn_state = tf.placeholder(tf.float32, [1, 256])

      # input -> S1L4
      h_fc1 = tf.nn.relu(tf.matmul(s_flat, self.W_fc1) + self.b_fc1)
      
      # S1L4 -> S1L236
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

      h_fc_in = tf.concat([h_fc1, h_fc2], 1)
      
      h_fc_in_reshaped = tf.reshape(h_fc_in, [1,-1,512])

      rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                      h_fc_in_reshaped,
                                                      initial_state = self.initial_rnn_state,
                                                      sequence_length = self.step_size,
                                                      time_major = False,
                                                      scope = scope)
      
      # rnn_outputs: (1, unroll_steps, 256) for back prop, (1,1,256) for forward prop
      
      rnn_outputs = tf.reshape(rnn_outputs, [-1,256])
      
      self.pi = tf.nn.softmax(tf.matmul(rnn_outputs, self.W_fc3) + self.b_fc3)
      v_ = tf.matmul(rnn_outputs, self.W_fc4) + self.b_fc4
      self.v = tf.reshape( v_, [-1] )
      
      scope.reuse_variables()
      self.W_rnn0 = tf.get_variable("basic_rnn_cell/weights0")
      self.b_rnn0 = tf.get_variable("basic_rnn_cell/biases0")
      self.W_rnn1 = tf.get_variable("basic_rnn_cell/weights1")
      self.b_rnn1 = tf.get_variable("basic_rnn_cell/biases1")
      
      self.reset_state()

  def reset_state(self):
    self.rnn_state_out = np.zeros([1, 256])
    
  def run_policy_and_value(self, sess, s_t, last_action):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.rnn_state_out = sess.run( [self.pi, self.v, self.rnn_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_rnn_state : self.rnn_state_out,
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_value(self, sess, s_t, last_action):
    prev_rnn_state_out = self.rnn_state_out
    v_out, _ = sess.run( [self.v, self.rnn_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_rnn_state : self.rnn_state_out,
                                      self.step_size : [1]} )
    # roll back rnn state
    self.rnn_state_out = prev_rnn_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_fc1, self.b_fc1,
            self.W_rnn0, self.b_rnn0,
            self.W_rnn1, self.b_rnn1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_fc4, self.b_fc4]


class SomaticActionRecurrentNetwork(SomaticNetwork):
  """
  Add action input for M2.
  """
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    SomaticNetwork.__init__(self, action_size, thread_index, device)
    scope_name = "net_" + str(self._thread_index)
    
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_fc1, self.b_fc1 = self._fc_variable([100, 256]) #

      self.W_fc2, self.b_fc2 = self._fc_variable([256, 256]) # 
      
      # weight for policy output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc4, self.b_fc4 = self._fc_variable([256, 1])

      # rnn cell
      self.cell = SomaticActionRNNCell(256, action_size)

      # place holder for RNN unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      # Last action
      self.last_action_input = tf.placeholder(tf.float32, [None, self._action_size])
    
      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])
      s_flat = tf.reshape(self.s, [-1, 100])

      self.initial_rnn_state = tf.placeholder(tf.float32, [1, 256])

      # input -> S1L4
      h_fc1 = tf.nn.relu(tf.matmul(s_flat, self.W_fc1) + self.b_fc1)
      
      # S1L4 -> S1L236
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      
      h_fc_in = tf.concat([h_fc1, h_fc2, self.last_action_input], 1)
      
      h_fc_in_reshaped = tf.reshape(h_fc_in, [1,-1,515])

      rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                      h_fc_in_reshaped,
                                                      initial_state = self.initial_rnn_state,
                                                      sequence_length = self.step_size,
                                                      time_major = False,
                                                      scope = scope)
      
      # rnn_outputs: (1, unroll_steps, 256) for back prop, (1,1,256) for forward prop
      
      rnn_outputs = tf.reshape(rnn_outputs, [-1,256])
      
      self.pi = tf.nn.softmax(tf.matmul(rnn_outputs, self.W_fc3) + self.b_fc3)
      v_ = tf.matmul(rnn_outputs, self.W_fc4) + self.b_fc4
      self.v = tf.reshape( v_, [-1] )
      
      scope.reuse_variables()
      self.W_rnn0 = tf.get_variable("basic_rnn_cell/weights0")
      self.b_rnn0 = tf.get_variable("basic_rnn_cell/biases0")
      self.W_rnn1 = tf.get_variable("basic_rnn_cell/weights1")
      self.b_rnn1 = tf.get_variable("basic_rnn_cell/biases1")
      
      self.reset_state()

  def reset_state(self):
    self.rnn_state_out = np.zeros([1, 256])
      
  def run_policy_and_value(self, sess, s_t, last_action):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    last_a = np.zeros([ACTION_SIZE])
    last_a[last_action] = 1
    
    pi_out, v_out, self.rnn_state_out = sess.run( [self.pi, self.v, self.rnn_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.last_action_input : [last_a],
                                                                self.initial_rnn_state : self.rnn_state_out,
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_value(self, sess, s_t, last_action):    
    prev_rnn_state_out = self.rnn_state_out

    last_a = np.zeros([ACTION_SIZE])
    last_a[last_action] = 1
    
    v_out, _ = sess.run( [self.v, self.rnn_state],
                         feed_dict = {self.s : [s_t],
                                      self.last_action_input : [last_a],
                                      self.initial_rnn_state : self.rnn_state_out,
                                      self.step_size : [1]} )
    # roll back rnn state
    self.rnn_state_out = prev_rnn_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_fc1, self.b_fc1,
            self.W_rnn0, self.b_rnn0,
            self.W_rnn1, self.b_rnn1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_fc4, self.b_fc4]


class SomaticCNNNetwork(SomaticNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    SomaticNetwork.__init__(self, action_size, thread_index, device)
    scope_name = "net_" + str(self._thread_index)
    
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([3, 3, 1, 8],  "base_conv1")
      self.W_conv2, self.b_conv2 = self._conv_variable([3, 3, 8, 16], "base_conv2")
      
      self.W_fc1, self.b_fc1 = self._fc_variable([16, 256]) #
      
      # weight for policy output layer
      self.W_fc2, self.b_fc2 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 2) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
      h_conv2_flat = tf.reshape(h_conv2, [-1, 16])
      
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
      v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
      self.v = tf.reshape( v_, [-1] )
      
  def run_policy_and_value(self, sess, s_t, last_action):
    pi_out, v_out = sess.run( [self.pi, self.v], feed_dict = {self.s : [s_t]} )
    return (pi_out[0], v_out[0])
                                           
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_value(self, sess, s_t, last_action):
    v_out = sess.run( self.v, feed_dict = {self.s : [s_t]} )
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3]

  
class SomaticRecurrentCNNNetwork(SomaticNetwork):
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device="/cpu:0" ):
    SomaticNetwork.__init__(self, action_size, thread_index, device)
    scope_name = "net_" + str(self._thread_index)
    
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      self.W_conv1, self.b_conv1 = self._conv_variable([3, 3, 1, 8],  "base_conv1")
      self.W_conv2, self.b_conv2 = self._conv_variable([3, 3, 8, 16], "base_conv2")
      
      self.W_fc1, self.b_fc1 = self._fc_variable([16, 256]) #      
      self.W_fc2, self.b_fc2 = self._fc_variable([256, 256]) # 
      
      # weight for policy output layer
      self.W_fc3, self.b_fc3 = self._fc_variable([256, action_size])

      # weight for value output layer
      self.W_fc4, self.b_fc4 = self._fc_variable([256, 1])

      # rnn cell
      self.cell = SomaticRNNCell(256)

      # place holder for RNN unrolling time step size.
      self.step_size = tf.placeholder(tf.float32, [1])

      # state (input)
      self.s = tf.placeholder("float", [None, 10, 10, 1])

      h_conv1 = tf.nn.relu(self._conv2d(self.s,  self.W_conv1, 2) + self.b_conv1)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
      h_conv2_flat = tf.reshape(h_conv2, [-1, 16])
      
      self.initial_rnn_state = tf.placeholder(tf.float32, [1, 256])

      # input -> S1L4
      h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
      
      # S1L4 -> S1L236
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)

      h_fc_in = tf.concat([h_fc1, h_fc2], 1)
      
      h_fc_in_reshaped = tf.reshape(h_fc_in, [1,-1,512])

      rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                      h_fc_in_reshaped,
                                                      initial_state = self.initial_rnn_state,
                                                      sequence_length = self.step_size,
                                                      time_major = False,
                                                      scope = scope)
      
      # rnn_outputs: (1, unroll_steps, 256) for back prop, (1,1,256) for forward prop
      
      rnn_outputs = tf.reshape(rnn_outputs, [-1,256])
      
      self.pi = tf.nn.softmax(tf.matmul(rnn_outputs, self.W_fc3) + self.b_fc3)
      v_ = tf.matmul(rnn_outputs, self.W_fc4) + self.b_fc4
      self.v = tf.reshape( v_, [-1] )
      
      scope.reuse_variables()
      self.W_rnn0 = tf.get_variable("basic_rnn_cell/weights0")
      self.b_rnn0 = tf.get_variable("basic_rnn_cell/biases0")
      self.W_rnn1 = tf.get_variable("basic_rnn_cell/weights1")
      self.b_rnn1 = tf.get_variable("basic_rnn_cell/biases1")
      
      self.reset_state()

  def reset_state(self):
    self.rnn_state_out = np.zeros([1, 256])
    
  def run_policy_and_value(self, sess, s_t, last_action):
    # This run_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.rnn_state_out = sess.run( [self.pi, self.v, self.rnn_state],
                                                   feed_dict = {self.s : [s_t],
                                                                self.initial_rnn_state : self.rnn_state_out,
                                                                self.step_size : [1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  def run_value(self, sess, s_t, last_action):
    prev_rnn_state_out = self.rnn_state_out
    v_out, _ = sess.run( [self.v, self.rnn_state],
                         feed_dict = {self.s : [s_t],
                                      self.initial_rnn_state : self.rnn_state_out,
                                      self.step_size : [1]} )
    # roll back rnn state
    self.rnn_state_out = prev_rnn_state_out
    return v_out[0]

  def get_vars(self):
    return [self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_rnn0, self.b_rnn0,
            self.W_rnn1, self.b_rnn1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_fc4, self.b_fc4]
