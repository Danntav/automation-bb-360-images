"""
Automated Annotation Pipeline for 360° Images (Equirectangular)
"""

import numpy as np
import cv2
import math
from omnicv import fisheyeImgConv
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology


result = {}
base_model = GroundingDINO(ontology=CaptionOntology({"car": "car"}))
mapper = fisheyeImgConv()


def get_user_inputs():
    """
    Collects scan configurations.
    """
    
    pitch = int(input("Pitch step value (in degrees): "))
    yaw = int(input("Yaw step value (in degress): "))
    fov_input = int(input("FoV value in degrees (default=90): ") or 90)
 
    return pitch, yaw, fov


def extract_views(image, pitch, yaw, h, w, fov=90):
    """
    Extracts a pinhole perspective from the original equirectangular image
    based on the given pitch and yaw angles. It triggers detection.
    """
    
    print(f"Pitch: {pitch}, Yaw: {yaw}, FoV: {fov}")
    persp = mapper.eqruirect2persp(image, fov, yaw, pitch, h, w)
    image_name = "img_" + str(pitch) + "_" + str(yaw) + ".jpg"
    item_detection(persp, pitch, yaw)


def item_detection(imagem, pitch, yaw):
    """
    Performs Zero-Shot detection in plan view using GroundingDINO.
    """
    coordinates = base_model.predict(imagem)
    coordinates = coordinates[coordinates.confidence > 0.55]
    result[(pitch, yaw)] = (coordinates.xyxy, coordinates.confidence)


def pixel_projection(x, y, pitch, yaw, image_H, image_W, H=512, W=512, fov=math.pi/2):
    """
    Performs the reverse projection of a pixel from the local 2D plane of the virtual camera 
    for the global coordinates of the 360 ​​equirectangular image.
    
    Math:
    1. Moves the origin to the optical center.
    2. Calculates focal length.
    3. Assemble the local vector in 3D space
    4. Apply the rotation matrices (Pitch on the X axis, Yaw on the Y axis) 
    5. Normalizes the vector and converts it to spherical coordinates (latitude and longitude).
    6. Maps the sphere back to the equirectangular matrix (pixels).
    """

    # Pitch and yaw in rads
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    
    # Offset to the image's center
    cx = W / 2
    cy = H / 2
        
    x_pixel = x - cx
    y_pixel = y - cy
    
    # Focal distance
    f = (W / 2) / math.tan(fov / 2)
    
    # Vector Rotation
    vlocal = np.array([x_pixel, y_pixel, f]).T
    
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])

    vglobal = Ry @ Rx @ vlocal
    
    # Normalizing
    norm = np.linalg.norm(vglobal)
    v_norm = vglobal / norm
    x_g, y_g, z_g = v_norm
    
    # 3D Coordinates -> XY Coordinates
    long = math.atan2(x_g, z_g)
    lat = math.asin(y_g)
    
    screen_x = (long + math.pi) * (image_W / (2 * math.pi))
    screen_y = (lat + math.pi / 2) * (image_H / math.pi)
    
    return int(screen_x), int(screen_y)


def scan_spherical_image(image, step_pitch, step_yaw, height, width, fov):
    """
    Orchestrates the virtual cameras across the 360 image to extract views.
    """
    
    for current_pitch in np.arange(-90, 90, step_pitch):
        for current_yaw in np.arange(0, 360, step_yaw):
            extract_views(image, current_pitch, current_yaw, height, width, fov)


def convert_to_global_coord(result_dict, image_h, image_w):
    """
    Projects local coordinates back to the global 360 map.
    To avoid distortions and curvatures on the BBs we extract the 4 corners of the original BB,
    use the min() and max() of these coordinates to draw a new BB rectangular and aligned to the
    axes of the flat picture.
    """
    
    local_boxes = []
    local_scores = []
    
    for (pitch, yaw), (box_list, score_list) in result_dict.items():
        for bbox, score in zip(box_list, score_list):
            xmin, ymin, xmax, ymax = bbox
            
            p_tl = (xmin, ymin) # Top-Left
            p_tr = (xmax, ymin) # Top-Right
            p_bl = (xmin, ymax) # Bottom-Left
            p_br = (xmax, ymax) # Bottom-Right
            
            # 4 corners of the box in the equirectangular image
            x1, y1 = pixel_projection(p_tl[0], p_tl[1], pitch, yaw, image_h, image_w)
            x2, y2 = pixel_projection(p_tr[0], p_tr[1], pitch, yaw, image_h, image_w)
            x3, y3 = pixel_projection(p_bl[0], p_bl[1], pitch, yaw, image_h, image_w)
            x4, y4 = pixel_projection(p_br[0], p_br[1], pitch, yaw, image_h, image_w)
            
            xmin_global = min(x1, x2, x3, x4)
            xmax_global = max(x1, x2, x3, x4)
            ymin_global = min(y1, y2, y3, y4)
            ymax_global = max(y1, y2, y3, y4)
            
            # Anti Pac-Man Filter (Cross-border issue)
            if (xmax_global - xmin_global) > (image_w / 2):
                continue
                
            local_boxes.append([xmin_global, ymin_global, xmax_global - xmin_global, ymax_global - ymin_global])
            local_scores.append(float(score))
            
    return local_boxes, local_scores


def apply_nms(all_boxes, all_scores, score_threshold=0.5, nms_threshold=0.4):
    """
    Applies Non-Maximum Suppression to filter overlapping boxes. Returns a clean list.
    """
    
    filtered_boxes = []
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, score_threshold=score_threshold, nms_threshold=nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append(all_boxes[i])
            
    return filtered_boxes


def render_and_save_boxes(image, boxes, output_filename="result.jpg"):
    """
    Draws the final bounding boxes on the image and saves it.
    """
    
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imwrite(output_filename, image)
    print(f"Image saved as '{output_filename}'")
    

def main():
    image_path = "360.jpg"
    image = cv2.imread(image_path)
    
    image_h, image_w, _ = image.shape
    height, width = 512, 512

    # Get config
    step_pitch, step_yaw, fov = get_user_inputs()

    # Scan image and detect objects
    scan_spherical_image(image, step_pitch, step_yaw, height, width, fov)
    
    # Project local coordinates back to the global 360 map
    all_boxes, all_scores = convert_to_global_coord(result, image_h, image_w)
    
    # NMS filter
    final_boxes = apply_nms(all_boxes, all_scores)
    
    # Save
    render_and_save_boxes(image, final_boxes)

if __name__ == "__main__":
    main()
