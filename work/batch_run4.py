# coding=utf-8

# write code...
import subprocess

def run(cmd):
    subprocess.check_call("python3 {}".format(cmd) , shell= True)

if __name__ == "__main__":

    run("run_preprocess4_usefamilysize.py")
    run("run_lgbm4.py")

