import cv2
import numpy as np
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

def find_board_vEllipse(img_path):

    img = cv2.imread(img_path)
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    w, h = img.shape[:2]
    scale_factor = 0.25
    resized_img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
    
    imCalHSV = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite('images/ellipse/resized-HSV.jpg', imCalHSV)
    kernel = np.ones((5, 5), np.float32) / 25
    blur = cv2.filter2D(imCalHSV, -1, kernel)
    cv2.imwrite('images/ellipse/resized-blur.jpg', blur)
    h, s, imCal = cv2.split(blur)

    ret, thresh = cv2.threshold(imCal, 140, 255, cv2.THRESH_BINARY_INV)

    # removes border wire outside the outerellipse
    kernel = np.ones((5, 5), np.uint8)
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,kernel)
    cv2.imwrite('images/ellipse/resized-thresh2.jpg', thresh2)
    # find enclosing ellipse
    ellipse, img_with_contour = findEllipseNew(thresh2 , resized_img)
    #print("center of the ellipse: ", ellipse[0])
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL) 

    return ellipse, img

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

#gen
def findEllipseOG(thresh2,img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_output = cv2.Canny(img_gray, thresh2, thresh2 * 2)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ellipse = None
    for i, c in enumerate(contours):
        if c.shape[0] > 5:
            ellipse = cv2.fitEllipse(c)
            break
    (x, y), (MA, ma), angle = ellipse
    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    return ellipse, img

def findEllipseNew(thresh2, img):

    canny_output = auto_canny(thresh2, sigma=2)
    cv2.imshow("autocanny", canny_output)
    cv2.waitKey()
    edges = canny(thresh2, sigma = 2.0, low_threshold=0.55, high_threshold=0.8)

    cv2.waitKey()
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("nr of contours: ", len(contours))
    largest_ellipse = None
    largest_size = 0
    ellipses = []

    for i, c in enumerate(contours):
        if c.shape[0] > 5:
            ellipse = cv2.fitEllipse(c)
            ellipses.append(ellipse)
            (x, y), (MA, ma), angle = ellipse  # Extract ellipse parameters
            size = MA * ma  # Calculate the size (area approximation)
                
            # Update the largest ellipse if the current one is larger
            if size > largest_size:
                largest_size = size
                largest_ellipse = ellipse
            print("found ellipse nr: ", i)
        
    if largest_ellipse is None:
        raise ValueError("No ellipse found")
    else:
        # Specify color (e.g., bright green) and thickness (e.g., 2 pixels)
        color = (255, 0, 0)  # Bright green
        thickness = 2        # 2 pixels thick
        
        # Draw the ellipse on the image
        cv2.ellipse(img, largest_ellipse, color, thickness)

        for ellipse in ellipses:
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        # Display the image with the ellipse
        cv2.imshow("Image with Ellipse", img)
        cv2.waitKey()
        cv2.destroyAllWindows() 
    return largest_ellipse, img



def findEllipse(binary_img, original_img):
    """
    Find the most prominent ellipse in a binary image.
    
    Args:
        binary_img (np.ndarray): Binary image where the object is white on black background
        original_img (np.ndarray): Original image for visualization purposes
    
    Returns:
        tuple: (ellipse object, annotated original image)
            ellipse object has attributes: center, size, angle
    """
    # Make a copy of the images to avoid modifying originals
    binary = binary_img.copy()
    img_with_contours = original_img.copy()
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Filter contours based on area to remove noise
    min_area = 1000  # Adjust this threshold based on your image size
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not valid_contours:
        raise ValueError("No valid contours found after area filtering")
    
    # Find the largest contour by area
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Fit an ellipse to the largest contour
    # The contour must have at least 5 points to fit an ellipse
    if len(largest_contour) < 5:
        raise ValueError("Contour has too few points to fit an ellipse")
    
    try:
        ellipse = cv2.fitEllipse(largest_contour)
        
        # Validate ellipse parameters
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
        
        # Check if the ellipse dimensions make sense
        if major_axis < 10 or minor_axis < 10:  # Adjust these thresholds
            raise ValueError("Ellipse too small - likely spurious detection")
        
        # Check if the aspect ratio is reasonable for a dartboard
        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
        if aspect_ratio > 1.5:  # Dartboards should be roughly circular
            raise ValueError("Ellipse aspect ratio too high - likely not a dartboard")
        
        # Draw the contour for visualization
        cv2.drawContours(img_with_contours, [largest_contour], -1, (0, 0, 255), 2)
        
        return ellipse, img_with_contours
    
    except cv2.error:
        raise ValueError("Failed to fit ellipse to contour")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', default='images/testboard.jpg')
    args = parser.parse_args()
    find_board_vEllipse(args.input)