# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class SomaticRNNCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state):
    with tf.variable_scope("basic_rnn_cell") as scope:
      p0, p1 = tf.split(value=inputs, num_or_size_splits=2, axis=1)
      p0_size = p0.get_shape()[1]
      p1_size = p1.get_shape()[1]

      output_size = self.output_size

      # [L4 + M2からのリカレント]-> [L5]への重み
      weights0 = tf.get_variable("weights0",
                                [p0_size + output_size, output_size],
                                dtype=tf.float32)
      biases0 = tf.get_variable("biases0",
                               [output_size],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))

      # [L2/3,6] -> [M2] への重み
      weights1 = tf.get_variable("weights1",
                                [p1_size + output_size, output_size],
                                dtype=tf.float32)
      biases1 = tf.get_variable("biases1",
                               [output_size],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))

      # L4からの出力とM2からのリカレントをconcatして、L5へ入れる
      p0_concat = tf.concat([p0, state], 1)
      output    = tf.tanh(tf.nn.bias_add(tf.matmul(p0_concat, weights0), biases0))
      
      # L5からの出力と、L23,6から出力をconcatして、M2へ入れる
      p1_concat = tf.concat([p1, output], 1)
      state_out = tf.tanh(tf.nn.bias_add(tf.matmul(p1_concat,  weights1), biases1))
    return output, state_out



class SomaticActionRNNCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, action_size):
    self._num_units = num_units
    self._action_size = action_size

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state):
    with tf.variable_scope("basic_rnn_cell") as scope:
      # inputs = (1,512)
      
      p0, p1 = tf.split(value=inputs, num_or_size_splits=[256,256+self._action_size], axis=1)
      
      # p0 = (1,256)
      # p1 = (1,256+action_size)
      
      print("p0.shape=",p0.get_shape()) #..
      print("p1.shape=",p1.get_shape()) #..
      
      p0_size = p0.get_shape()[1]
      p1_size = p1.get_shape()[1]
      
      output_size = self.output_size
      
      # [L4 + M2からのリカレント]-> [L5]への重み
      weights0 = tf.get_variable("weights0",
                                [p0_size + output_size, output_size],
                                dtype=tf.float32)
      biases0 = tf.get_variable("biases0",
                               [output_size],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))
      
      # [L2/3,6] -> [M2] への重み
      weights1 = tf.get_variable("weights1",
                                [p1_size + output_size, output_size],
                                dtype=tf.float32)
      biases1 = tf.get_variable("biases1",
                               [output_size],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))
      
      # L4からの出力とM2からのリカレントをconcatして、L5へ入れる
      p0_concat = tf.concat([p0, state], 1)
      output    = tf.tanh(tf.nn.bias_add(tf.matmul(p0_concat, weights0), biases0))
      
      # L5からの出力と、L23,6から出力をconcatして、M2へ入れる
      p1_concat = tf.concat([p1, output], 1)
      state_out = tf.tanh(tf.nn.bias_add(tf.matmul(p1_concat,  weights1), biases1))
    return output, state_out
