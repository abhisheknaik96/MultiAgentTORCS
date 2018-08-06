import threading
import multiprocessing
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import playGame_DDPG
#matplotlib inline
import os
from random import choice
from time import sleep
from time import time
import snakeoil3_gym as snakeoil3
#import pymp

#class Worker(object):
#    def __init__(self, name, port):

with tf.device("/cpu:0"): 
        num_workers = 6 #multiprocessing.cpu_count()
        print("numb of workers is" + str(num_workers))
        #workers = []
        #for i in range(num_workers):
        #        client = snakeoil3.Client(p=3101+i, vision=False)  # Open new UDP in vtorcs
        #        client.MAX_STEPS = np.inf
        #        workers.append("")#playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=1, port=3101+i))

with tf.Session() as sess:
        worker_threads = []
#with pymp.Parallel(4) as p:
        for i in range(num_workers):
                worker_work = lambda: (playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=0, port=3101+i))
                print("hi i am here \n")
                t = threading.Thread(target=(worker_work))
                print("active thread count is: " + str(threading.active_count()) + "\n")
                t.start()
                sleep(0.5)
                worker_threads.append(t)
