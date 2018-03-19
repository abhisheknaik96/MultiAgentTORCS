import os, time

os.system('torcs -nofuel -nodamage -nolaptime -vision &')
time.sleep(0.5)
os.system('sh autostart.sh')
time.sleep(0.5)