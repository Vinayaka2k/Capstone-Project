from subprocess import call
import os

a = os.listdir("./trained_gan")

for i in a:
    print(i.split(".")[0])