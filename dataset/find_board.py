import cv2
import numpy as np
import argparse
import os
import os.path as osp
import random

def find_board_vEllipse2(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    """approach 1: (failes in warm lighting)"""
    red_channel, green_channel, blue_channel = cv2.split(rgb)
    redMinusGray = cv2.subtract(red_channel, gray, -1, None) # red mask
    greenMinusGray = cv2.subtract(green_channel, gray, -1, None) # green mask
    red_binary = otsu_thresholding(redMinusGray) # get binary image
    green_binary = otsu_thresholding(greenMinusGray) # get binary image
    raw_binary = cv2.bitwise_or(red_binary, green_binary) # combine red and green masks
    resize_binary, scale_factor = proportionalResize(raw_binary, 1000) # resize to 1000px max side

    """approach 2: (hardcoded color ranges)"""
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    final_mask = cv2.bitwise_or(red_mask, green_mask)
    approach2_binary = otsu_thresholding(final_mask)

    resize_app2_binary, scale_factor = proportionalResize(approach2_binary, 1000) # resize to 1000px max side

    closed_binary = cv2.morphologyEx(resize_app2_binary, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) # close the gaps

    largestEllipse = findEllipse(closed_binary) # find the largest ellipse
    if largestEllipse is None:
        print("No ellipse found.")
        return None
    ellipseCenter, axes, angle = largestEllipse
    
    bbox = getSquareBboxForEllipse(ellipseCenter, axes, closed_binary.shape[0], 
                                  closed_binary.shape[1])
    resized_bbox = (int(bbox[0][0]/scale_factor), int(bbox[0][1]/scale_factor)), \
                   (int(bbox[1][0]/scale_factor), int(bbox[1][1]/scale_factor))
    size_of_resized_bbox = (resized_bbox[1][0] - resized_bbox[0][0]) * (resized_bbox[1][1] - resized_bbox[0][1])
    print("size of bbox = ", size_of_resized_bbox)
    reformatted_bbox = [resized_bbox[0][1], resized_bbox[1][1], 
                       resized_bbox[0][0], resized_bbox[1][0]]
    

    #these lines are for visualization purposes only
    #resized_ellipse = (ellipseCenter[0]/scale_factor, ellipseCenter[1]/scale_factor), (axes[0]/scale_factor, axes[1]/scale_factor), angle
    cv2.circle(img, (int(ellipseCenter[0]/scale_factor), int(ellipseCenter[1]/scale_factor)), 5, (0, 255, 0), -1)
    cv2.rectangle(img, resized_bbox[0], resized_bbox[1], (0, 255, 0), 10)
    demo_resize = cv2.resize(img, (0, 0), fx = 0.2, fy = 0.2)
    cv2.imshow("Bounding box preview:", demo_resize)
    cv2.waitKey()

    return reformatted_bbox

def otsu_thresholding(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, binary_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary_img

def proportionalResize(image, target_size, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    max_side = max(h, w)
    
    if max_side > target_size:
        r = target_size / float(max_side)
        return cv2.resize(image, (0, 0), fx=r, fy=r), r
    return image, 1


def findEllipse(img, original_img=None, scale_factor=1):
    """
    input: binary image
    output: params of roated rectangle around ellipse
    center point, axes (long and short), angle
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    best_ellipse = None

    for c in contours:
        if c.shape[0] > 5:
            ellipse = cv2.fitEllipse(c)
            (x, y), (MA, ma), angle = ellipse
            size = MA * ma
            circularity = MA / ma
            ellipses.append((ellipse, size, circularity))

    if len(ellipses) == 0:
        return None
    else:
        ellipses.sort(key=lambda x: x[1], reverse=True)

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:
            best_ellipse = ellipse[0] # for now its just the largest ellipse with circularity > 0.8
            break
    if best_ellipse is None:
        best_ellipse = ellipses[0][0] # if no ellipse with circularity > 0.8, return the largest ellipse
    return best_ellipse

def getSquareBboxForEllipse(center, axes, h, w):
    """
    input: center point, axes (long and short), image height and width
    output: top left and bottom right corners of a square bounding box with 
    the same center as the ellipse and 25-35% larger than the ellipse
    """
    scale_factor = 1 + random.uniform(0.3, 0.35) # outer rim is 32% larger than the ellipse
    long_axis = max(axes) * scale_factor # set the bounding box to be 25-35% larger than the ellipse
   
    top_left_corner = (max(0,int(center[0] - long_axis/2)), 
                      max(0, int(center[1] - long_axis/2)))
    bottom_right_corner = (min(w, int(center[0] + long_axis/2)), 
                          min(h,int(center[1] + long_axis/2)))
    
    return (top_left_corner, bottom_right_corner)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Board detection using ellipse fitting.')
    parser.add_argument('-i', '--input', help='Path to input image.', 
                       default='images/boards/myboard/approach1-failed.jpg')
    args = parser.parse_args()
    
    if osp.isdir(args.input):
        for img in os.listdir(args.input):
            if not osp.isdir(osp.join(args.input, img)):
                print("bbox = ", find_board_vEllipse2(osp.join(args.input, img)))
    else:
        print("bbox = ", find_board_vEllipse2(args.input))