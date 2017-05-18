# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from environment import Environment
from model import *

from constants import *

LOG_INTERVAL = 100
PERFORMANCE_LOG_INTERVAL = 1000

class Trainer(object):
  def __init__(self,
               thread_index,
               model_type,
               env_type,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.model_type = model_type
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.local_network = create_model(thread_index, model_type)
    
    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
    
    self.env = Environment(env_type)
    self.last_state = self.env.reset()
    self.last_action = 0
    
    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

  @property
  def use_rnn(self):
    return self.model_type == "rnn" or \
      self.model_type == "rnn_action" or \
      self.model_type == "rnn_cnn"

  @property
  def use_action_input(self):
    return self.model_type == "rnn_action"

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)

  def _record_score(self, sess, summary_writer,
                    summary_op, score_input, step_size_input,
                    score, step_size,
                    global_t):
    
    summary_str = sess.run(summary_op, feed_dict={
      score_input    : score,
      step_size_input: step_size
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input, step_size_input):
    states = []
    actions = []
    rewards = []
    values = []
    last_actions = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if self.use_rnn:
      start_rnn_state = self.local_network.rnn_state_out
    
    # t_max times loop
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess,
                                                            self.last_state,
                                                            self.last_action)
      action = self.choose_action(pi_)

      states.append(self.last_state)
      actions.append(action)
      values.append(value_)
      last_actions.append(self.last_action)
      
      self.last_action = action

      if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
        print("pi={}".format(pi_))
        print(" V={}".format(value_))

      # process game
      state, reward, terminal, _ = self.env.step(action)

      self.episode_reward += reward

      # clip reward
      rewards.append( reward )

      self.local_t += 1

      if terminal:
        terminal_end = True
        print("score={} step={}".format(self.episode_reward, self.env.step_size))

        self._record_score(sess, summary_writer,
                           summary_op, score_input, step_size_input,
                           self.episode_reward, self.env.step_size,
                           global_t)
          
        self.episode_reward = 0
        self.last_state = self.env.reset()
        self.last_action = 0
        if self.use_rnn:        
          self.local_network.reset_state()
        break

    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, state, self.last_action)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()
    last_actions.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []
    batch_last_a = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi, last_ai) in zip(actions, rewards, states, values, last_actions):
      R = ri + GAMMA * R
      td = R - Vi
      
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1
      
      last_a = np.zeros([ACTION_SIZE])
      last_a[last_ai] = 1
      
      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_last_a.append(last_a)
      batch_R.append(R)
      
    cur_learning_rate = self._anneal_learning_rate(global_t)
    
    batch_si.reverse()
    batch_a.reverse()
    batch_td.reverse()
    batch_R.reverse()
    batch_last_a.reverse()
    
    feed_dict = {
      self.local_network.s: batch_si,
      self.local_network.a: batch_a,
      self.local_network.td: batch_td,
      self.local_network.r: batch_R,
      self.learning_rate_input: cur_learning_rate }
    
    if self.use_rnn:
      rnn_feed_dict = {
        self.local_network.initial_rnn_state: start_rnn_state,
        self.local_network.step_size : [len(batch_a)] }
      feed_dict.update(rnn_feed_dict)
    
    if self.use_action_input:
      action_input_feed_dict = {
        self.local_network.last_action_input: batch_last_a }
      feed_dict.update(action_input_feed_dict)
    
    sess.run( self.apply_gradients, feed_dict=feed_dict )
    
    if (self.thread_index == 0) and (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
      self.prev_local_t += PERFORMANCE_LOG_INTERVAL
      elapsed_time = time.time() - self.start_time
      steps_per_sec = global_t / elapsed_time
      print("### Performance : {} STEPS in {:.0f} sec. {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
        global_t,  elapsed_time, steps_per_sec, steps_per_sec * 3600 / 1000000.))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
