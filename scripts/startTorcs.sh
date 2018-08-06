#!/bin/bash
while true;
do
	ps cax | grep torcs > /dev/null
	if [ $? -eq 0 ]; then
	  : #echo "Process is running."
	else
	  echo "Process is not running."
	  cd /home/meha/vtorcs2 && ./torcs & sh autostart.sh
          	  	  
	fi
done;


