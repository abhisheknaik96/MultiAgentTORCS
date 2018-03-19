# Multi-agent TORCS for RL
---

This is a multi-agent version of TORCS, for multi-agent reinforcement learning. In other words, the multiple cars running simultaneously on a track can be controlled by different control algorithms - heuristic, reinforcement learning-based, etc.


## Dependencies

- TORCS 		(the simulator)
- scr-client 	(the patch which creates a server-client model)

---

## Installation

It is assumed that you have TORCS installed from the source code on a machine with Ubuntu 14.04/16.04 LTS.

### scr-client

Install the scr-client as follows:

1.	Download the scr-patch from [here](https://sourceforge.net/projects/cig/files/SCR%20Championship/Server%20Linux/2.1/scr-linux-patch.tgz/download).
2.	Unpack the package scr-linux-patch.tgz in your base TORCS directory.
3.	This will create a new directory called scr-patch.     
    ```
    cd scr-patch
    ```
4.	`sh do_patch.sh` (`do_unpatch.sh` to revert the modifications)     
5.	Move to the parent TORCS directory    
    `cd ../`
6.	Run the following commands
    ```
    ./configure    
    make -j4    
    sudo make install -j4    
    sudo make datainstall -j4    
    ```

10 scr_server car should be available in the race configurations now.

7.	Download the C++ client from [here](https://sourceforge.net/projects/cig/files/SCR%20Championship/Client%20C%2B%2B/2.0/).
8.	Unpack the package `scr-client-cpp.tgz` in your base TORCS directory.
9.	This will create a new directory called `scr-client-cpp`.     
    `cd scr-client-cpp`
10.	`make -j4`
11.	At this point, multiple clients can join an instance of the TORCS game by:
    ```
    ./client    
    ./client port:3002
    ```
	Typical values are between 3001 and 3010 (3001 is the default)


---

## Usage:

1. 	Start a 'Quick Race' in TORCS in one terminal console (with the n agents being `scr_*`)
..*	`$ torcs`
2. 	From inside the multi-agent-torcs directory in one console:
..*	`$ python playGame.py 3001`
3. 	From another console:
..*	`$ python playGame.py 3002`
	And so on...

In the game loop in `playGame.py`, the action at every timestep `a_t` can be supplied by any algorithm.

---

## Acknowledgements

The multi-agent learning simulator was developed by [Abhishek Naik](http://abhisheknaik96.github.io/), under the guidance of Anirban Santara, Balaraman Ravindran, and Bharat Kaul.