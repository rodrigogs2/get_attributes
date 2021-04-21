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


