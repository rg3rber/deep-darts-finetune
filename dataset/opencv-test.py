from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import time
from skimage import transform
from skimage.feature import canny

rng.seed(12345)
def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    
    #old version it returns 3 values: _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv.drawContours(drawing, contours, i, color)
        # ellipse
        if c.shape[0] > 5:
            cv.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = cv.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(drawing, [box], 0, color)
    
    
    cv.imshow('Contours', drawing)


def change_hue(val):
    global hue_thresh
    hue_thresh = val
    color_thresh_callback()


def change_saturation(val):
    global sat_thresh
    sat_thresh = val
    color_thresh_callback()

def change_value(val):
    global val_thresh
    val_thresh = val
    color_thresh_callback()

def color_thresh_callback():
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    # Green range
    lower_green = np.array([hue_thresh, sat_thresh, val_thresh])
    upper_green = np.array([80, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)

    cv.imshow('Color Mask', green_mask)
    
    

def detect_dartboard_vColorSegmentation(img):
    """
    Detect dartboard in img using color segmentation followed by ellipse fitting.
    
    Args:
        img_path: Path to the input img file
    Returns:
        tuple: (original img with detection overlay, detected ellipse parameters)
    """
    
    # Convert to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define color ranges for dartboard colors
    lower_red1 = np.array([0, 120, 70]) 
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 60, 40]) # initial: 40,40,40 improved?: 40, 40-60, 40 for outdoor: needs more sat 60 and val 80-100.
    upper_green = np.array([80, 255, 255])
    
    
    # Create masks for each color
    red_mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv.bitwise_or(red_mask1, red_mask2)
    green_mask = cv.inRange(hsv, lower_green, upper_green)
    
    # Combine masks
    combined_mask = cv.bitwise_or(red_mask, green_mask)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img, None
    
    # Find the largest contour that can be fit with an ellipse
    valid_contours = [cnt for cnt in contours if len(cnt) >= 5]
    if not valid_contours:
        return img, None
    
    # Sort by area and try to fit ellipses starting with largest
    valid_contours.sort(key=cv.contourArea, reverse=True)
    
    result_img = img.copy()
    detected_ellipse = None
    
    for contour in valid_contours:
        try:
            # Calculate contour features
            area = cv.contourArea(contour)
            if area < 1000:  # Skip tiny contours
                continue
                
            ellipse = cv.fitEllipse(contour)
            
            # Calculate ellipse parameters
            center, axes, angle = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else float('inf')
            
            # Filter out bad ellipses
            if aspect_ratio > 2.5:  # Too elongated
                continue
            if aspect_ratio < 1.1:  # Too circular (might be inner circle)
                continue
                
            # Draw the ellipse
            cv.ellipse(result_img, ellipse, (0, 255, 0), 3)
            detected_ellipse = ellipse
            
            # Draw center point
            center = tuple(map(int, center))
            cv.circle(result_img, center, 5, (0, 0, 255), -1)
            
            # Found a good ellipse, stop looking
            break
            
        except cv.error:
            continue
    
    # Debug visualization
    cv.imshow('Color Mask', combined_mask)
    cv.imshow('Result', result_img)
    cv.waitKey()
    cv.destroyAllWindows()
    
    return result_img, detected_ellipse

def benchmark_color_approach(image, iterations=10):
    times = []
    for i in range(iterations):
        print("iteration: ", i)
        start = time.time()
        
        # Color segmentation
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        red_mask1 = cv.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        red_mask2 = cv.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        green_mask = cv.inRange(hsv, np.array([40, 40, 40]), np.array([80, 255, 255]))
        combined_mask = cv.bitwise_or(cv.bitwise_or(red_mask1, red_mask2), green_mask)
        
        # Find contours and fit ellipse
        contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv.contourArea)
            if len(largest) >= 5:
                cv.fitEllipse(largest)
                
        times.append(time.time() - start)
    return np.mean(times)

def benchmark_hough_approach(image, iterations=10):
    times = []
    for i in range(iterations):
        print("iteration: ", i)
        start = time.time()
        
        # Edge detection
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = canny(img_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
        print("did canny")
        # Hough ellipse detection
        result = transform.hough_ellipse(edges, accuracy=20, threshold=250, min_size=100)
        
        times.append(time.time() - start)
    return np.mean(times)


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
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

    return cv.resize(image, dim, interpolation=inter)

def find_board_vHoughCircles(img):

    h, w = img.shape[:2]

    scale = 0.5
    img_resized = cv.resize(img, (int(w * scale), int(h * scale)))
    print("resized: ", img_resized.shape)
    
    # Convert to grayscale.
    gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY) 
     
    gray_blurred = cv.GaussianBlur(gray, (31,31), cv.BORDER_DEFAULT)


    minR = round(w/40)
    maxR = round(w/5)
    minDis = round(w/5)

    print("min_radius: ", minR)
    print("max_radius: ", maxR)
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv.HoughCircles(
        gray_blurred,  
        cv.HOUGH_GRADIENT, 
        1.3, 
        minDist = minDis, 
        param1 = 25, 
        param2 = 60,
        minRadius = minR, 
        maxRadius = maxR
    )
    
    # Draw circles that are detected. 
    if detected_circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        print("found circles: ", detected_circles.shape)
        
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
    
            # Draw the circumference of the circle. 
            cv.circle(img_resized, (a, b), r, (0, 255, 0), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv.circle(img_resized, (a, b), 1, (0, 0, 255), 3)
            #resizeImg = ResizeWithAspectRatio(img_resized, width=1000)
            cv.imshow("Detected Circle", img_resized) 
            cv.waitKey()
            


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', default='images/testboard.jpg')
    args = parser.parse_args()
    src = cv.imread(cv.samples.findFile(args.input))
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    #find_board_vHoughCircles(src)
    
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
    max_thresh = 255
    thresh = 100 # initial threshold
    cv.createTrackbar('hue:', source_window, thresh, max_thresh, change_hue)
    cv.createTrackbar('saturation:', source_window, thresh, max_thresh, change_saturation)
    cv.createTrackbar('value:', source_window, thresh, max_thresh, change_value)

    color_thresh_callback()
    cv.waitKey()
    cv.destroyAllWindows()

    """
    color_time = benchmark_color_approach(src)
    hough_time = benchmark_hough_approach(src)
    print(f"Color approach average time: {color_time*1000:.2f}ms")
    print(f"Hough approach average time: {hough_time*1000:.2f}ms")

    """

    """opencv example
    # Convert img to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    src_gray = cv.blur(src_gray, (3,3))
    source_window = 'Source'
    cv.namedWindow(source_window)
    cv.imshow(source_window, src)
    max_thresh = 255
    thresh = 200 # initial threshold
    cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh)
    cv.waitKey()
     """

    """
    try:
        result_img, ellipse = detect_dartboard(src)
        if ellipse is None:
            print("No dartboard detected!")
        else:
            center, axes, angle = ellipse
            print(f"Dartboard detected at center: {center}")
            print(f"Major axis: {max(axes):.1f}, Minor axis: {min(axes):.1f}")
            print(f"Rotation angle: {angle:.1f} degrees")
            
            # Save result
            #output_path = 'detected_dartboard.jpg'
            #cv.imwrite(output_path, result_img)
            #print(f"Result saved to {output_path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        """
    



   