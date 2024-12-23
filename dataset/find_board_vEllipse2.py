import cv2
import numpy as np
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt


def find_board_vEllipse2(img_path):

    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print('Could not open or find the image:', img_path)
        exit(0)
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blue_channel, green_channel, red_channel = cv2.split(rgb)
    hue_channel, saturation_channel, value_channel = cv2.split(hsv)

    # Now you can access the intensities of the red and green channels:
    # The "red_channel" contains the red intensities
    # The "green_channel" contains the green intensities

    # Example: Display the red and green channel intensities
    cv2.imshow('Red Channel', red_channel)
    cv2.imshow('Green Channel', green_channel)
    

    # Optionally, you can save the individual channels as separate images
    cv2.imwrite('images/delaney/red_channel.jpg', red_channel)
    cv2.imwrite('images/delaney/green_channel.jpg', green_channel)
    cv2.imwrite('images/delaney/blue_channel.jpg', blue_channel)
    cv2.imwrite('images/delaney/hue_channel.jpg', hue_channel)
    cv2.imwrite('images/delaney/saturation_channel.jpg', saturation_channel)
    cv2.imwrite('images/delaney/value_channel.jpg', value_channel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print("gray shape: ", gray.shape)
    print("red shape: ", red_channel.shape)
    print("green shape: ", green_channel.shape)
    print("blue shape: ", blue_channel.shape)
    mask = None
    dtype = -1
    grayMinusRed = cv2.subtract(gray, red_channel, dtype, mask)
    cv2.imwrite('images/delaney/grayMinusRed.jpg', grayMinusRed)
    
    grayMinusGreen = cv2.subtract(gray, green_channel, dtype, mask)
    cv2.imwrite('images/delaney/grayMinusGreen.jpg', grayMinusGreen)
   
    grayMinusBlue = cv2.subtract(gray, blue_channel, dtype, mask)
    cv2.imwrite('images/delaney/grayMinusBlue.jpg', grayMinusBlue)






    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

 
    # Otsu's thresholding from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret3,th3 = cv2.threshold(blur,255,0,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ret1,th1 = cv2.threshold(blur,140,255,cv2.THRESH_BINARY)

    print("Global threshold value: ", ret1)
    print("Otsu's threshold value: ", ret2)
    print("Otsu's threshold value after Gaussian filtering: ", ret3)

    # plot all the images and their histograms
    images = [gray, 0, th1,
            gray, 0, th2,
            blur, 0, th3]
    
    cv2.imwrite('images/delaney/gray.jpg', gray)
    cv2.imwrite('images/delaney/blur.jpg', blur)
    
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=80)',
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
     
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
    parser.add_argument('-i', '--input', help='Path to input image.', default='images/testboard.jpg')
    args = parser.parse_args()
    find_board_vEllipse2(args.input)