import metaworld
import random
import time
import numpy as np

from termcolor import colored

print(colored(metaworld.ML1.ENV_NAMES,"magenta",attrs=["reverse","bold"]))  # Check out the available environments


#['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2']

#if 1:
    #bn_name="pick-place-v2"
    #bn_name="assembly-v2"
    #bn_name="faucet-open-v2"

action_spaces=[]
obs_spcaes=[]
for bn_name in metaworld.ML1.ENV_NAMES:

    print(bn_name)
    #bn_name="basketball-v2"
    #bn_name="hammer-v2"
    #bn_name="assembly-v2"
    #bn_name="pick-place-v2"
    #bn_name="hand-insert-v2"
    bn_name="button-press-topdown-v2"
    bn_name="plate-slide-back-v2"
    bn_name="sweep-v2"
    bn_name="hand-insert-v2"
    bn_name="disassemble-v2"
    bn_name="soccer-v2"

    ml1 = metaworld.ML1(bn_name) #constructs the benchmark which is an environment. As this is ML1, only the task (i.e. the goal)
                                 #will vary. So ml1.train_classes is going to be of lenght 1
    
    print(colored(ml1.train_classes,"blue",attrs=["reverse"]))
    
    env = ml1.train_classes[bn_name]()  
    task = random.choice(ml1.train_tasks)#changes goal

    env.set_task(task)  # Set task
    
    obs = env.reset()  # Reset environment

    action_spaces.append(env.action_space.sample().shape)
    obs_spcaes.append(env.observation_space.sample().shape)


    #print(colored(f"obj_init_angle, obj_init_pos, goal=={env.obj_init_angle}   ,  {env.obj_init_pos}    ,  {env.goal}", "red"))
    print(colored(f"obj_init_pos, goal=={env.obj_init_pos}    ,  {env.goal}", "red"))
    if 1:
        #for step in range(env.max_path_length):
        for step in range(300):
            #print("curr_path_length==",env.curr_path_length)
            env.render()
            a = env.action_space.sample()  # Sample an action
            #a = [1,1,1,0.1]
            #a = [1,1,1,1]
            a=[0,0,0,0.1]
            #print(obs[:4])
            #print(colored(env._get_site_pos('rightEndEffector'),"red"))
            #print(colored(env._get_site_pos('leftEndEffector'),"red"))
            #print(colored(np.linalg.norm(env._get_site_pos('rightEndEffector')-env._get_site_pos('leftEndEffector')),"red"))

            obs, reward, done, info = env.step(a)
            time.sleep(1/60)

    env.close()
