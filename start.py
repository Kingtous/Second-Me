import os
import subprocess
import webbrowser
import threading



current_executable_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_executable_dir)

PYTHON_EXE = os.path.join(current_executable_dir, "env", "python.exe")
NODE_EXE =  os.path.join(current_executable_dir, "node", "node.exe")
NPM_BIN = os.path.join(current_executable_dir, "node", "node_modules", "npm", "bin", "npm-cli.js")


server_cmd = [PYTHON_EXE, "-m", "flask",  "run", "--host", "0.0.0.0", "--port=8002"]
# dev 模式跑起来
fronted_cmd = [NODE_EXE, NPM_BIN, "run",  "dev"]



PATHS_APPEND = [
    os.path.join(current_executable_dir, "env"),
    os.path.join(current_executable_dir, "env", "Scripts"),
    # node
    os.path.join(current_executable_dir, "node"),
]

# add paths to system path
for path in PATHS_APPEND:
    os.environ["PATH"] = path + ";" + os.environ["PATH"]

# ENV PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app \
#     BASE_DIR=/app/data \
#     LOCAL_LOG_DIR=/app/logs \
#     RUN_DIR=/app/run \
#     RESOURCES_DIR=/app/resources \
#     APP_ROOT=/app \
#     FLASK_APP=lpm_kernel.app
os.environ['PYTHONUNBUFFERED'] = "1"
os.environ['PYTHONPATH'] = os.path.join(current_executable_dir, "lpm_kernel")
os.environ['BASE_DIR'] = os.path.join(current_executable_dir, "data")
os.environ['LOCAL_LOG_DIR'] = os.path.join(current_executable_dir, "logs")
os.environ['RUN_DIR'] = os.path.join(current_executable_dir, "run")
os.environ['RESOURCES_DIR'] = os.path.join(current_executable_dir, "resources")
os.environ['APP_ROOT'] = current_executable_dir
os.environ['FLASK_APP'] = "lpm_kernel.app"
os.environ['PYTHONUTF8'] = "1"  # set python utf8 mode
os.environ['PYTHONEXE'] = PYTHON_EXE
os.environ['LLAMACPPSERVERBIN'] = os.path.join(current_executable_dir, "llama.cpp", "llama-server.exe")
# add python to path

def print_log(p):
    while True:
        line = p.stdout.readline()
        if line == b"":
            continue
        print("SERVER", line)
        if b"Error" in line:
            break

def run_server_wait():
    p = subprocess.Popen(args=server_cmd , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        line = p.stdout.readline()
        if line == b"":
            continue
        print(line)
        if b"Running on" in line:
            threading.Thread(target=print_log, args=(p,), daemon=True).start()
            break
    return p

def run_fronted():
    p = subprocess.Popen(args=fronted_cmd, stdout=subprocess.PIPE, cwd=os.path.join(current_executable_dir, "lpm_frontend"), stderr=subprocess.STDOUT)
    # wait for server to start, "Wait stdout Running on"
    while True:
        line = p.stdout.readline()
        if line == b"":
            continue
        print(line)
        if b"Ready in" in line:
            break
    return p

def test_graphrag():
    p = subprocess.Popen(args=['graphrag', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    for line in p.stderr:
        print("ERR:", line.strip())
    for line in p.stdout:
        print("LINE:", line.strip())
    p.wait()
    print(p.returncode)
        
if __name__ == "__main__":
    try:
        test_graphrag()
        sp = run_server_wait()
        # sp.wait()
        p = run_fronted()
        # open browser
        # webbrowser.open("http://localhost:3000")
        p.wait()
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
        
        
    
    