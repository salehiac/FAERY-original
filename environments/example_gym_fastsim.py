#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    05/06/2019
""" 

import gym, gym_fastsim
import time

display= True

env = gym.make('FastsimSimpleNavigation-v0')
env.reset()
action=[10,11]

if(display):
	env.enable_display()

then = time.time()

for i in range(10000):
        env.render()
        o,r,eo,info=env.step(action)
        print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(o), r, info["dist_obj"], str(info["robot_pos"]), str(eo)))
        if(display):
        	time.sleep(0.01)
        if eo:
            break

now = time.time()

print("%d timesteps took %f seconds" % (i, now - then))

env.close()
