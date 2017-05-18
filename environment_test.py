# -*- coding: utf-8 -*-
import unittest
import numpy as np

from environment import Environment
import scipy.misc
import scipy

class TestSequenceFunctions(unittest.TestCase):

  def test_create_rough_image(self):
    environment = Environment()
    
    image = environment.create_rough_image(50, 10)
    
    self.assertTrue( image.shape == (10, 50) )
    self.assertTrue( np.amax(image) <= 1.0 )
    self.assertTrue( np.amin(image) >= 0.0 )

    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save("rough.png")

    
  def test_create_smooth_image(self):
    environment = Environment()

    image = environment.create_smooth_image(50, 10)
    
    self.assertTrue( image.shape == (10, 50) )
    self.assertTrue( np.amax(image) <= 1.0 )
    self.assertTrue( np.amin(image) >= 0.0 )

    scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save("smooth.png")

    
  def test_reset(self):
    environment = Environment()
    sensor_image0 = environment.reset()
    self.assertTrue( sensor_image0.shape == (10, 10, 1) )


  def test_action(self):
    environment = Environment()

    # ザラザラ地面で左端に到達すると正リワード
    environment.reset()
    environment.rough_floor = True

    for i in range(20):
      sensor_image1, reward, terminal, _ = environment.step(2)

    self.assertTrue( sensor_image1.shape == (10, 10, 1) )
    self.assertTrue( reward == 1 )
    self.assertTrue( terminal )

    # ザラザラ地面で右端に到達すると負リワード 
    environment.reset()
    environment.rough_floor = True
    
    for i in range(20):
      sensor_image1, reward, terminal, _ = environment.step(1)
    self.assertTrue( reward == -1 )
    self.assertTrue( terminal )

    # つるつる地面で左に到達すると負リワード
    environment.reset()
    environment.rough_floor = False

    for i in range(20):
      sensor_image1, reward, terminal, _ = environment.step(2)
    self.assertTrue( reward == -1 )
    self.assertTrue( terminal )

    # つるつる地面で右に到達すると負リワード
    environment.reset()
    environment.rough_floor = False

    for i in range(20):
      sensor_image1, reward, terminal, _ = environment.step(1)
    self.assertTrue( reward == 1 )
    self.assertTrue( terminal )

    # つるつる地面で右に到達すると負リワード
    environment.reset()
    environment.rough_floor = False
    
    # 停止アクションの確認
    for i in range(20):
      sensor_image1, reward, terminal, _ = environment.step(0)
    self.assertTrue( environment.pos == 25 )
    self.assertTrue( reward == 0 )
    self.assertFalse( terminal )

  def test_random_check(self):
    environment = Environment()

    rough_count = 0
    smooth_count = 0

    for i in xrange(100):
      environment.reset()
      if environment.rough_floor:
        rough_count += 1
      else:
        smooth_count += 1
    print("rough={}, smooth={}".format(rough_count, smooth_count))
    

if __name__ == '__main__':
  unittest.main()
