# coding=utf-8

# write code...
import subprocess

def run(cmd):
    subprocess.check_call("python3 {}".format(cmd) , shell= True)

if __name__ == "__main__":

    run("run_preprocess3_usecabin.py")
    run("run_lgbm3.py")

