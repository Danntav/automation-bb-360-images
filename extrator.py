##implementar PyVips
import numpy as np
import cv2
import math
import sys
from tqdm import tqdm
from omnicv import fisheyeImgConv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

result = {}

def extrair_vistas(image, pitch, yaw, h, w, fov=90):
	print(f"Yaw: {yaw}, pitch: {pitch}, fov: {fov}")
	persp=mapper.eqruirect2persp(image, fov, pitch, yaw, h, w)
	image_name = "img_" + str(pitch) + "_" + str(yaw) + ".jpg"
	cv2.imwrite(image_name, persp)

image = cv2.imread("360.jpg")
#cv2.imshow("Display image", image)
#cv2.waitKey(0)

salto_yaw = int(input("yaw value: "))
salto_pitch = int(input("pitch value: "))
fov = int(input("fov value: (default=90)") or 90)

#height, width, _ = image.shape

height, width = 512,512

mapper = fisheyeImgConv()

#for current_pitch in tqdm(np.arange(-math.pi/2, math.pi/2, salto_pitch)):
for current_pitch in tqdm(np.arange(-90, 90, salto_pitch)):
	for current_yaw in np.arange(0, 360, salto_yaw):
		extrair_vistas(image, current_pitch, current_yaw, height, width, fov)

