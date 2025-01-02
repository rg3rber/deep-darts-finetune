import cv2
import numpy as np
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd
import time

def find_board_vEllipse2(img_path):

    img = cv2.imread(img_path)
    
    if img is None:
        print('Could not open or find the image:', img_path)
        exit(0)

    img_name=osp.splitext(osp.basename(img_path))[0]
    folder_name="images/delaney"
 
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    red_channel, green_channel, blue_channel = cv2.split(rgb)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv)

    """ # Example: Display the red and green channel intensities
    cv2.imshow('Red Channel', red_channel)
    cv2.imshow('Green Channel', green_channel)
    
    cv2.imshow('hsv', hsv)
    cv2.waitKey()
    cv2.imshow('hue Channel', hue_channel)
    cv2.waitKey()
    cv2.imshow('saturation Channel', saturation_channel)
    cv2.waitKey()
    cv2.imshow('value Channel', value_channel)
    cv2.waitKey()
    return """

    # Optionally, you can save the individual channels as separate images
    """ cv2.imwrite('images/delaney/rndm-red_channel.jpg', red_channel)
    cv2.imwrite('images/delaney/rndm-green_channel.jpg', green_channel)
    cv2.imwrite('images/delaney/rndm-blue_channel.jpg', blue_channel)
    cv2.imwrite('images/delaney/rndm-hue_channel.jpg', hue_channel)
    cv2.imwrite('images/delaney/rndm-saturation_channel.jpg', saturation_channel)
    cv2.imwrite('images/delaney/rndm-value_channel.jpg', value_channel)
 """

    mask = None
    dtype = -1
    """ grayMinusRed = cv2.subtract(gray, red_channel, dtype, mask)
    cv2.imwrite('images/delaney/rndm-grayMinusRed.jpg', grayMinusRed)
    
    grayMinusGreen = cv2.subtract(gray, green_channel, dtype, mask)
    cv2.imwrite('images/delaney/rndm-grayMinusGreen.jpg', grayMinusGreen)
   
    grayMinusBlue = cv2.subtract(gray, blue_channel, dtype, mask)
    cv2.imwrite('images/delaney/rndm-grayMinusBlue.jpg', grayMinusBlue) """

    
    redMinusGray = cv2.subtract(red_channel, gray, dtype, mask)
    greenMinusGray = cv2.subtract(green_channel, gray, dtype, mask)
    blueMinusGray = cv2.subtract(blue_channel, gray, dtype, mask)

    #writing difference images:
    """ 
    cv2.imwrite(osp.join(folder_name, img_name+'-redMinusGray.jpg'), redMinusGray)
    
    cv2.imwrite(osp.join(folder_name, img_name+'-greenMinusGray.jpg'), greenMinusGray)
    
    cv2.imwrite(osp.join(folder_name, img_name+'-blueMinusGray.jpg'), blueMinusGray) """

    red_binary = otsu_thresholding(redMinusGray)
    green_binary = otsu_thresholding(greenMinusGray)

    #combine the two masks:
    raw_binary = cv2.bitwise_or(red_binary, green_binary)
    # cv2.imshow('Combined Mask raw', raw_binary)
    # cv2.waitKey()

    resize_binary, scale_factor = proportionalResize(raw_binary, 1000)
    resize_gray, scale_factor = proportionalResize(gray, 1000)
    """ weird approach: grayEdgeDrawing = cv2.ximgproc.EdgeDrawing(resize_gray)
    detected_edges = cv2.ximgproc.EdgeDrawing.detectEdges(grayEdgeDrawing)

    detected_ellipses = cv2.ximgproc.EdgeDrawing.detectEllipses(detected_edges) """

    """ for ellipse in detected_ellipses:
        cv2.ellipse(resize_binary, ellipse, (0, 255, 0), 2)
    cv2.imshow('All Detected Ellipses', resize_binary) """

    kernel = np.ones((5, 5), np.uint8)
    closed_binary = cv2.morphologyEx(resize_binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    #closed_binary_kernel = cv2.morphologyEx(resize_binary, cv2.MORPH_CLOSE,kernel)

    largestEllipse = findEllipse(closed_binary, img_name)
    
    #read out ellipse parameters
    ellipseCenter, axes, angle = largestEllipse

    resized_ellipse = (ellipseCenter[0]/scale_factor, ellipseCenter[1]/scale_factor), (axes[0]/scale_factor, axes[1]/scale_factor), angle
    #cv2.ellipse(img, resized_ellipse, (0, 255, 0), 2)

    """ cv2.circle(closed_binary, (int(ellipseCenter[0]), int(ellipseCenter[1])), 5, (0, 255, 0), -1)

    cv2.imshow("Image with center marked", closed_binary)
    cv2.waitKey() """

    bbox = getSquareBboxForEllipse(ellipseCenter, axes, closed_binary.shape[0], closed_binary.shape[1])
    #cv2.rectangle(closed_binary, (int(bbox[0][0]), int(bbox[0][1] )), (int(bbox[1][0]), int(bbox[1][1])), (0, 255, 0), 10)
    #cv2.imshow("small with BBOX", closed_binary)
    #cv2.waitKey()

    #* resized box has structure: [(x1, y1), (x2, y2)]
    resized_bbox = (int(bbox[0][0]/scale_factor), int(bbox[0][1]/scale_factor)), (int(bbox[1][0]/scale_factor), int(bbox[1][1]/scale_factor))
    #deep-darts needs the bbox in the format: bbox = [y1 y2 x1 x2]
    reformatted_bbox = [resized_bbox[0][1], resized_bbox[1][1], resized_bbox[0][0], resized_bbox[1][0]]

    
    cv2.circle(img, (int(ellipseCenter[0]/scale_factor), int(ellipseCenter[1]/scale_factor)), 5, (0, 255, 0), -1)

    cv2.rectangle(img, resized_bbox[0], resized_bbox[1], (0, 255, 0), 2)
    demo_resize = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
    cv2.imshow("Bounding box preview:", demo_resize)
    cv2.waitKey()
    
    #store_path = osp.join('images/boards/bbox/myboard', img_name+'-bbox.jpg')
    #print("Storing image at: ", store_path)
    #cv2.imwrite(store_path, img)

    return reformatted_bbox

def otsu_thresholding(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    threshold_value ,binary_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary_img


def otsu_thresholding_plt(img):
    # Otsu's thresholding from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ret1,th1 = cv2.threshold(blur,140,255,cv2.THRESH_BINARY)

    print("Global threshold value: ", ret1)
    print("Otsu's threshold value: ", ret2)
    print("Otsu's threshold value after Gaussian filtering: ", ret3)

    # plot all the images and their histograms
    images = [blur, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    
    """   #cv2.imwrite('images/delaney/rndm-gray.jpg', gray)
    #cv2.imwrite('images/delaney/rndm-blur.jpg', blur) """
    
    titles = ['Gaussian Filtered Image','Histogram','Global Thresholding (v=140)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    
    y_ticks = plt.gca().get_yticks()

    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        #plot histograms: 
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks(np.arange(0, 256, 10))
          # Get the current y-ticks
        plt.yticks(np.arange(0, max(y_ticks), 10))
        plt.xlabel('Pixel Intensity ')  # X-axis label
        plt.ylabel('Pixels Frequency')  # Y-axis label

        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


#target size will be the longer side of the image
def proportionalResize(image, target_size, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    max_side = max(h, w)
    if max_side > target_size: #only resize if the image is larger than the target size
        r = target_size / float(max_side)
        return cv2.resize(image, (0, 0), fx = r, fy = r), r

    return image, 1

def findEllipse(img, contours_log=None, img_name=None):

    #using canny:
    """ canny_output = auto_canny(img)
    canny_contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of canny contours found: ", len(canny_contours))
    path = osp.join('images/boards/bbox/', '_'+ img_name)
    cv2.imwrite(path + 'canny.jpg', canny_output)
    cv2.imwrite(path + 'binary.jpg', img)
    print("Stored canny and binary image at: ", path) """
    
    #directly using the binary image:
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #maybe try different retrieval mode and contour approximation method:
    #contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    largest_ellipse = None
    largest_size = 0
    ellipses = []

    start = time.time() #TODO remove
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
    end = time.time()
    if largest_ellipse is None:
        raise ValueError("No ellipse found")
    else:
        # Specify color (e.g., bright green) and thickness (e.g., 2 pixels)
        color = (255, 0, 0)  # Bright green
        thickness = 2        # 2 pixels thick
        
        # Draw the ellipse on the image
        #cv2.ellipse(img, largest_ellipse, color, thickness)

        #for ellipse in ellipses:
        #    cv2.ellipse(img, ellipse, (0, 255, 0), 2)
        # Display the image with the ellipse
    
    #for testing: contours_log.loc[len(contours_log)] = [img_name, len(contours), end-start]

    return largest_ellipse

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def getSquareBboxForEllipse(center, axes, h, w):

    long_axis = max(axes)*1.1 #add 10% to the long axis
   
    top_left_corner = (max(0,int(center[0] - long_axis/2)), max(0, int(center[1] - long_axis/2)))
    bottom_right_corner = (min(w, int(center[0] + long_axis/2)), min(h,int(center[1] + long_axis/2)))
    # Calculate the bounding box
    bbox = (top_left_corner, bottom_right_corner)
    return bbox

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', default='images/boards/myboard/testboard.jpg')
    args = parser.parse_args()
    curr_path = args.input
    contours_log = pd.DataFrame(columns=['image', 'nr of contours', 'time to enumerate contours'])
    if osp.isdir(curr_path):
        for img in os.listdir(curr_path):
                if not osp.isdir(osp.join(curr_path, img)):
                    find_board_vEllipse2(osp.join(curr_path, img))
        #dump contours as txt
        contours_log.to_csv(osp.join(curr_path,'contours_log.csv'), index=False)
    else:
        find_board_vEllipse2(curr_path)

    