import os
import cv2
import sys
import numpy as np
from skimage.transform import resize

sys.path.append("../")
ROOT_DIR = os.path.abspath(os.curdir)

from fire_evacuation.symbols_to_obstacles import add_obstacles_to_GAN
from fire_evacuation.image_boundary import get_final_mask

RED_LOWER_THRES = [(0, 50, 50), (20, 255, 255)]
RED_UPPER_THRES = [(150, 50, 50), (180, 255, 255)]
BLUE_THRES = [(80, 50, 50), (150, 255, 255)]
GREEN_THRES = [(35, 50, 50), (85, 255, 255)]
BLACK_THRES = [(0, 0, 0), (0, 0, 0)]

def add_border_img(img):
    bordersize = 50
    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    
    return border

def down_sample_venue(img):
    height, width = img.shape[:2]
    while height > 150 and width > 150:
        scale_percent = 75  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = resize(img, dim, order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        height, width = img.shape[:2]
    return(img)    


def color_threshold(img, thresholds=[]):
    result = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f"{ROOT_DIR}/input/images/hsv.png", img)
    masks = None
    for thres in thresholds:
        # Blue color
        lower = np.array(thres[0])
        upper = np.array(thres[1])
        mask = cv2.inRange(img, lower, upper)
        if masks is None:
            masks = mask
        else:
            masks = masks + mask

    masks = cv2.medianBlur(masks, 3)
    cv2.imwrite(f"{ROOT_DIR}/input/images/wall_mask.png", masks)
    result[masks != 255] = 255

    return result


def get_wall_image_layer(img):
    return color_threshold(
        img,
        [BLUE_THRES],
    )


def get_window_image_layer(img):
    return color_threshold(
        img,
        [GREEN_THRES],
    )


def get_wall_window_image_layer(img):
    return color_threshold(
        img,
        [BLUE_THRES, GREEN_THRES],
    )


def get_door_image_layer(img):
    return color_threshold(
        img,
        [
            RED_LOWER_THRES,
            RED_UPPER_THRES,
        ],
    )


def get_obstacle_image_layer(img):
    return color_threshold(
        img,
        [BLACK_THRES]
    )


def convert(GAN_file):
    combined_img = add_obstacles_to_GAN(GAN_file)

    combined_img = add_border_img(combined_img)

    walls = get_wall_image_layer(combined_img)
    windows = get_window_image_layer(combined_img)
    doors = get_door_image_layer(combined_img)
    obstacles = get_obstacle_image_layer(combined_img)

    combined_img = down_sample_venue(combined_img)
    height, width = combined_img.shape[:2]

    walls = resize(walls, (height, width), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
    doors = resize(doors, (height, width), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
    obstacles = resize(obstacles, (height, width), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
    windows = resize(windows, (height, width), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')

    txt_floorplan = np.zeros(combined_img.shape[:2], 'U1')
    txt_floorplan.fill('E')

    final_mask = get_final_mask(combined_img, GAN_file)

    txt_floorplan[np.where(final_mask == 0)[:2]] = "_"
    txt_floorplan[np.where(walls != (255, 255, 255))[:2]] = "W"
    txt_floorplan[np.where(doors != (255, 255, 255))[:2]] = "S"
    txt_floorplan[np.where(obstacles != (255, 255, 255))[:2]] = "F"
    txt_floorplan[np.where(windows != (255, 255, 255))[:2]] = "E"

    np.savetxt(f"{ROOT_DIR}/fire_evacuation/floorplans/test_floorplan.txt", txt_floorplan, fmt="%s")
