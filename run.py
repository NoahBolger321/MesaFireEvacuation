import cv2
import os
import glob
import argparse

ROOT_DIR = os.path.abspath(os.curdir)
print(ROOT_DIR)

parser = argparse.ArgumentParser(description='Image for simulation')
parser.add_argument('-i', '--image', type=str, help='The raw floorplan for the simulation')
args = parser.parse_args()

filename = args.image.split("/")[-1]

# read the image, and upload to /input/images directory
img = cv2.imread(args.image)

if os.path.isdir("input/images/"):
    files = glob.glob("input/images/")
    for f in files[1:]:
        os.remove(f)
else:
    os.makedirs("input/images/")

cv2.imwrite(f"input/images/{filename}", img)


import requests

filename = f"{ROOT_DIR}/input/images/{filename}"
myobj = {'file_path': filename}

x = requests.post("http://127.0.0.1:5000/run_GAN", json = myobj)

from fire_evacuation.server import server

# Starts a visual server with our model
server.launch()
