#import os, getop, 
import sys
import subprocess, shlex

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, encoding='utf8')
    process.stdout.flush()
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def main(argv):
	cmd_str = "gedit --help"

	#output = 
	run_command(cmd_str)
	#print("output from cmd is: ",output)



if __name__ == "__main__":    
    main(sys.argv)
