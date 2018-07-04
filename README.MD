# MADRaS - Multi-Agent DRiving Simulator

This is a multi-agent version of TORCS, for multi-agent reinforcement learning. In other words, the multiple cars running simultaneously on a track can be controlled by different control algorithms - heuristic, reinforcement learning-based, etc.


## Dependencies

- TORCS         (the simulator)
- Simulated Car Racing modules  (the patch which creates a server-client model to expose the higher-level game features to the learning agent)
- Python3 (all future development will be in Python3; an old Python2 branch also exists [here](https://github.com/abhisheknaik96/MultiAgentTORCS/tree/python2Version)) 

---

## Installation

It is assumed that you have TORCS installed (tested on [version 1.3.6](https://github.com/UWEcoCAR/car-simulator/tree/master/torcs-1.3.6)) from the source code on a machine with Ubuntu 14.04/16.04 LTS.

### scr-client

Install the scr-client as follows:

1.  Download the scr-patch from [here](https://sourceforge.net/projects/cig/files/SCR%20Championship/Server%20Linux/2.1/scr-linux-patch.tgz/download).
2.  Unpack the package scr-linux-patch.tgz in your base TORCS directory.
3.  This will create a new directory called scr-patch.     
    `cd scr-patch`
4.  `sh do_patch.sh` (`do_unpatch.sh` to revert the modifications)     
5.  Move to the parent TORCS directory    
    `cd ../`
6.  Run the following commands:
    ```
    ./configure    
    make -j4    
    sudo make install -j4    
    sudo make datainstall -j4    
    ```

10 scr_server car should be available in the race configurations now.

7.  Download the C++ client from [here](https://sourceforge.net/projects/cig/files/SCR%20Championship/Client%20C%2B%2B/2.0/).
8.  Unpack the package `scr-client-cpp.tgz` in your base TORCS directory.
9.  This will create a new directory called `scr-client-cpp`.     
    `cd scr-client-cpp`
10. `make -j4`
11. At this point, multiple clients can join an instance of the TORCS game by:
    ```
    ./client    
    ./client port:3002
    ```
    Typical values are between 3001 and 3010 (3001 is the default)


---

## Usage:

1.  Start a 'Quick Race' in TORCS in one terminal console (with the n agents being `scr_*`)    
    `torcs`    
    Close the TORCS window.
2.  From inside the multi-agent-torcs directory in one console:    
    `python3 playGame.py 3001`
3.  From another console:    
    `python3 playGame.py 3002`    
    And so on...

In the game loop in `playGame.py`, the action at every timestep `a_t` can be supplied by any algorithm.    

Note : 
1. `playGame_DDPG.py` has the code for a sample RL agent learning with the [DDPG algorithm](http://proceedings.mlr.press/v32/silver14.pdf), while `playGame.py` has a dummy agent which just moves straight at every timestep.
2. Headless rendering for multiple-agent learning is under development. Contributions and ideas would be greatly appreaciated! 
---

### For single-agent learning:

1.   Start a 'Quick Race' in TORCS in one terminal console. Choose only one `scr` car and as many as traffic cars as you want (preferably `chenyi*`<sup>1</sup>, since they're programmed to follow individual lanes at speeds low enough for the agent to learn to overtake)
2.  From inside the multi-agent-torcs directory in one console:    
    `python3 playGame_DDPG.py 3001`    
    or any other port.

Sample results for a DDPG agent learned to drive in traffic are [available here](https://goo.gl/piuQmg).       

---

Do check out [the wiki](https://github.com/abhisheknaik96/MultiAgentTORCS/wiki) for this project for in-depth information about TORCS and getting Deep (Reinforcement) Learning to work on it.

--- 

<sub>1 The `chenyi*` cars can be installed from [Princeton's DeepDrive project](http://deepdriving.cs.princeton.edu/), which also adds a few maps from training and testing the agents. The default cars in TORCS are all programmed heuristic racing agents, which do not serve as good stand-ins for 'traffic'. Hence, using chenyi's code is highly recommended. </sub>

## Credits

The multi-agent learning simulator was developed by [Abhishek Naik](http://abhisheknaik96.github.io), extending `ugo-nama-kun`'s [`gym-torcs`](https://github.com/ugo-nama-kun/gym_torcs), and `yanpanlau`'s [project](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html) under the guidance of Anirban Santara, Balaraman Ravindran, and Bharat Kaul, at Intel Labs.

### Contributors

We believe MADRaS will enable new and veteran researchers in academia and the industry to make the dream of fully autonomous driving a reality. Towards the same, we believe that unlike the closed-source secretive technologies of the big players, this project will enable the community to work towards this goal *together*, pooling in thoughts and resources to achieve this dream faster. Hence, we're highly appreciative of all sorts of contributions, big or small, from fellow researchers and users :
- [Meha Kaushik](https://github.com/MehaKaushik)
