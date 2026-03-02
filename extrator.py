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
boxes = []
def extrair_vistas(image, pitch, yaw, h, w, fov=90):
	print(f"Yaw: {yaw}, pitch: {pitch}, fov: {fov}")
	persp=mapper.eqruirect2persp(image, fov, yaw, pitch, h, w)
	image_name = "img_" + str(pitch) + "_" + str(yaw) + ".jpg"
	deteccao_items(persp, pitch, yaw)
	cv2.imwrite(image_name, persp)


def deteccao_items(imagem, pitch, yaw):
	coordenadas = base_model.predict(imagem)
	coordenadas = coordenadas[coordenadas.confidence > 0.55]
	result[(pitch, yaw)]=coordenadas.xyxy


def pixel_plano(x, y, pitch, yaw, image_H, image_W, H=512, W=512, fov=math.pi/2):
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    #pitch e yaw em rads   
    cx = W/2
    cy = H/2
    
    #offset para o meio da imagem
    x_pixel = x - cx
    y_pixel = y - cy
    
    #distancia focal
    f = (W/2) / math.tan(fov/2)
    
    vlocal = np.array([x_pixel, y_pixel, f]).T
    
    Rx = np.array([[1, 0, 0],
    [0, math.cos(pitch), -math.sin(pitch)],
    [0, math.sin(pitch), math.cos(pitch)]])
    
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
    [0, 1, 0],
    [-math.sin(yaw), 0, math.cos(yaw)]])

    vglobal = Ry @ Rx @ vlocal
    
    #normalizar vetor para ser menor que <=1
    norm = np.linalg.norm(vglobal)
    
    v_norm = vglobal / norm
    x_g, y_g, z_g = v_norm

    long = math.atan2(x_g, z_g)
    lat = math.asin(y_g)
    
    screenX = (long + math.pi) * (image_W / (2*math.pi))
    screenY = (-lat + math.pi/2) * (image_H / math.pi)
    
    return int(screenX), int (screenY)
    

base_model = GroundingDINO(ontology=CaptionOntology({"car": "car"}))

image = cv2.imread("360.jpg")
#cv2.imshow("Display image", image)
#cv2.waitKey(0)

salto_yaw = int(input("yaw value: "))
salto_pitch = int(input("pitch value: "))
fov = int(input("fov value: (default=90)") or 90)

image_H, image_W, _ = image.shape

height, width = 512,512

mapper = fisheyeImgConv()

#for current_pitch in tqdm(np.arange(-math.pi/2, math.pi/2, salto_pitch)):
for current_pitch in tqdm(np.arange(-90, 90, salto_pitch)):
	for current_yaw in np.arange(0, 360, salto_yaw):
		extrair_vistas(image, current_pitch, current_yaw, height, width, fov)

for (pitch,yaw), lista_boxes in result.items():
    for bbox in lista_boxes:
        xmin, ymin, xmax, ymax = bbox
        
        p_se = xmin, ymin
        p_sd = xmax, ymin
        p_ie = xmin, ymax
        p_id = xmax, ymax
        
        #4 cantos da caixa na imagem equirrectanguilar, cada um com x e y
        x1,y1 = pixel_plano(p_se[0], p_se[1], pitch, yaw, image_H, image_W)
        x2,y2 = pixel_plano(p_sd[0], p_sd[1], pitch, yaw, image_H, image_W)
        x3,y3 = pixel_plano(p_ie[0], p_ie[1], pitch, yaw, image_H, image_W)
        x4,y4 = pixel_plano(p_id[0], p_id[1], pitch, yaw, image_H, image_W)
        
        xmin_global = min(x1,x2,x3,x4)
        xmax_global = max(x1,x2,x3,x4)
        ymin_global = min(y1,y2,y3,y4)
        ymax_global = max(y1,y2,y3,y4)
        
        cv2.rectangle(image, (xmin_global, ymin_global), (xmax_global, ymax_global), (0, 255, 0), 3)

cv2.imwrite("result.jpg", image)
print("Image saved as 'result.jpg'")
