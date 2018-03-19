################################################################### 
# 
#	All the configurations are done here.
#	
#	1. Toggle the is_training flag to 0 to test a saved model.
#	2. epsilon_start is, as the name suggests, where the annealing epsilon starts from
#	3. total_explore is used as : epsilon -= 1/total_explore
#
################################################################### 

visualize_after = 5
is_training 	= 0

# total_explore  	= 300000.0
total_explore  	= 600000.0
max_eps 		= 500
max_steps_eps 	= 1000

wait_at_beginning 	= 0
initial_wait_period = 200		# to give the other cars a headstart of these many steps

epsilon_start  	= 0.5				# 
start_reward 	= -10000			# these need to be changed if restarting the playGame.py script

save_location = './'
# save_location = 'saved_networks_our_traffic_noBrakes_backup_370episodes'
# save_location = 'saved_networks_our_traffic_brakes_after230episodes/'
# save_location = 'saved_networks_our_traffic_noBrakes_backup_230episodes/'
# save_location = 'saved_networks_our_traffic_noBrakes/'
# save_location = 'saved_networks_scr_noTraffic/'				# test this playGame_old.py

torcsPort 	= 3001
configFile 	= '~/.torcs/config/raceman/quickrace.xml'
# configFile = '~/.torcs/config/raceman/practice.xml'