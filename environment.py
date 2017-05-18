# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import noise
import math

from constants import ACTION_SIZE

SENSOR_WIDTH = 10
SENSOR_HEIGHT = 10
MAX_SIDE_STEP = 20
IMAGE_WIDTH = (MAX_SIDE_STEP + SENSOR_WIDTH / 2) * 2
IMAGE_HEIGHT = SENSOR_HEIGHT

class Environment(object):
  def __init__(self, env_type=2):
    self.env_type = env_type

  def create_rough_image(self, w, h):
    if self.env_type == 0:
      # 周波数だけが違うノイズの難しいバージョン
      image = self.create_noise_image(w, h, 0.5)
      return image * 0.5 + 0.5
    if self.env_type == 1:
      # 振幅だけが違うノイズの簡単なバージョン
      image = self.create_noise_image(w, h, 0.5)
      return image * 0.5 + 0.5    
    elif self.env_type == 2:
      # 縞模様大の比較 (30:20)
      image = self.create_sin_image(w, h, 30)
      return image
    elif self.env_type == 3:
      # 縞模様中の比較 (20:10)
      image = self.create_sin_image(w, h, 20)
      return image    
    else:
      # 縞模様小の比較 (6:4)
      image = self.create_zebra_image(w, h, 6)
      return image
    
  def create_smooth_image(self, w, h):
    if self.env_type == 0:
      # 周波数だけが違うノイズの難しいバージョン
      image = self.create_noise_image(w, h, 0.1)
      return image * 0.5 + 0.5
    elif self.env_type == 1:
      # 振幅だけが違うノイズの簡単なバージョン
      image = self.create_noise_image(w, h, 0.5)
      return image * 0.5 + 0.1
    elif self.env_type == 2:
      # 縞模様大の比較 (30:20)
      image = self.create_sin_image(w, h, 20)
      return image
    elif self.env_type == 3:
      # 縞模様中の比較 (20:10)
      image = self.create_sin_image(w, h, 10)
      return image
    else:
      # 縞模様小の比較 (6:4
      image = self.create_sin_image(w, h, 4)
      return image

  def create_noise_image(self, w, h, a):
    image = np.empty([h, w], dtype=float)
    for x in xrange(w):
      for y in xrange(h):
        image[y][x] = noise.pnoise2(a*x, a*y)
    return image

  def create_zebra_image(self, w, h, interval):
    image = np.empty([h, w], dtype=float)
    for x in xrange(w):
      c = x % interval
      if c < interval/2:
        p = 1.0
      else:
        p = 0.0
      for y in xrange(h):
        image[y][x] = p
    return image

  def create_sin_image(self, w, h, period):
    start_theta = 3.141592 * 2.0 * random.random()
    image = np.empty([h, w], dtype=float)
    for x in xrange(w):
      theta = (3.141592 * 2.0 * float(x) / period) + start_theta
      p = math.sin(theta)
      p = p * 0.5 + 0.5
      for y in xrange(h):
        image[y][x] = p
    return image
  
  def get_sensor_image(self):
    sensor_image = self.floor_image[:,self.pos-5:self.pos+5]
    reshaped_sensor_image = np.reshape(sensor_image, (10, 10, 1))
    return reshaped_sensor_image
  
  def reset(self):
    choice = random.randint(0,1)
    if choice == 0:
      # Rough floor texture
      self.floor_image = self.create_rough_image(IMAGE_WIDTH, IMAGE_HEIGHT)
      self.rough_floor = True
      # Goal is right end
    else:
      # Smooth floor texture
      self.floor_image = self.create_smooth_image(IMAGE_WIDTH, IMAGE_HEIGHT)
      self.rough_floor = False
      # Goal is left end
      
    self.pos = IMAGE_WIDTH / 2
    self.step_size = 0
    return self.get_sensor_image()

  def step(self, action):
    """
    0 = stop
    1 = move right
    2 = move left
    """
    self.step_size += 1
    
    if action == 1:
      # Move right
      self.pos += 1
    elif action == 2:
      # Move left
      self.pos -= 1

    if self.pos == SENSOR_WIDTH/2:
      # Reached left end
      terminal = True
      if self.rough_floor:
        # Reached rough reward sucessfully
        reward = 1
      else:
        reward = -1
    elif self.pos == IMAGE_WIDTH - SENSOR_WIDTH/2:
      # Reached right end
      terminal = True
      if self.rough_floor:
        reward = -1
      else:
        # Reached smooth reward sucessfully
        reward = 1
    else:
      terminal = False
      reward = 0

    sensor_image = self.get_sensor_image()    
    return sensor_image, reward, terminal, ""
