import os
import subprocess

# Rename filename and duration
cwd = '/home/emre/Desktop/77ghz/CLI/Release'
os.environ["LD_LIBRARY_PATH"] = cwd  # + ':' + os.getcwd() # error code 127 when not executed

stop_cmd = './DCA1000EVM_CLI_Control stop_record cf.json'.split()

os.system("gnome-terminal 'ls'")  # opens a new terminal

pid = subprocess.check_output(['pgrep gnome-terminal'], shell=True)

cmd = subprocess.Popen(['kill', str(pid.decode())[:-1]], cwd=cwd, shell=False,
                       stdin=subprocess.PIPE, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # , check=True)
cmd.wait()

cmd = subprocess.Popen(stop_cmd, cwd=cwd, shell=False, stdin=subprocess.PIPE, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
cmd.wait()
print('Restart successful!')


