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

with tf.device("/cpu:0"): 
        num_workers = 6 #multiprocessing.cpu_count() #use this when you want to use all the cpus
        print("numb of workers is" + str(num_workers))

with tf.Session() as sess:
        worker_threads = []
#with pymp.Parallel(4) as p:		#uncomment this for parallelization of threads
        for i in range(num_workers):
                worker_work = lambda: (playGame_DDPG.playGame(f_diagnostics=""+str(i), train_indicator=0, port=3101+i))
                print("hi i am here \n")
                t = threading.Thread(target=(worker_work))
                print("active thread count is: " + str(threading.active_count()) + "\n")
                t.start()
                sleep(0.5)
                worker_threads.append(t)
