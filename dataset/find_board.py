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

    """approach 2: (hardcoded color ranges)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    blurred_final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
    approach2_binary = otsu_thresholding(blurred_final_mask)

    red_binary = otsu_thresholding(red_mask)

    resizeEllipse, scale_factor_ellipse = proportionalResize(approach2_binary, 1000)
    largestEllipse = findEllipse(resizeEllipse, img_path)

    cv2.ellipse(img, largestEllipse, (255, 0, 255), 10)
    cv2.imshow("ellipse", img)
    cv2.waitKey()
    
    if largestEllipse is None:
        print("No ellipse found.")
        return None
    ellipseCenter, axes, angle = largestEllipse

    rescaled_ellipseCenter = (int(ellipseCenter[0]/scale_factor_ellipse), int(ellipseCenter[1]/scale_factor_ellipse))
    rescaled_axes = (int(axes[0]/2/scale_factor_ellipse), int(axes[1]/2/scale_factor_ellipse))

    radius = int(np.mean(rescaled_axes)) # mean of long and short axes
    print("from ellipse inferred radius= ", radius)
    
    bbox = getSquareBboxForEllipse(ellipseCenter, axes, approach2_binary.shape[0], 
                                  approach2_binary.shape[1])
    resized_bbox = (int(bbox[0][0]/scale_factor_ellipse), int(bbox[0][1]/scale_factor_ellipse)), \
                   (int(bbox[1][0]/scale_factor_ellipse), int(bbox[1][1]/scale_factor_ellipse))
    
    bullseye_bbox = int((resized_bbox[1][0] - resized_bbox[0][0])/2.5)
    print("bbox length = ", bullseye_bbox)

    crop_image_toBullseye = red_binary[resized_bbox[0][1]+bullseye_bbox:resized_bbox[1][1]-bullseye_bbox, resized_bbox[0][0]+bullseye_bbox:resized_bbox[1][0]-bullseye_bbox]
    resize_crop_imageToEllipse, scale_factor_circle = proportionalResize(crop_image_toBullseye, 480)

    #contours_cropped, board_center, radius = find_boardCenterHough(resize_crop_imageToEllipse) #find_boardCenter(resize_crop_imageToEllipse)
    board_center, center_axes, center_angle = workingfindEllipse(resize_crop_imageToEllipse)
    
    #if contours_cropped is None or board_center is None or radius is None:
    if board_center is None:
        print("No board center found.")
    else:
        resize_red, scale_factor_red = proportionalResize(red_binary, 1000)
        remapped_center = (int(board_center[0]/scale_factor_circle + resized_bbox[0][0]+bullseye_bbox), int(board_center[1]/scale_factor_circle + resized_bbox[0][1]+bullseye_bbox))
        red_center = (int(remapped_center[0]*scale_factor_red), int(remapped_center[1]*scale_factor_red))
        rotate_red_9 = rotate_image(resize_red, red_center, 18)
        red_board = cv2.bitwise_or(rotate_red_9, resize_red)
        cv2.imshow("red", red_board)
        cv2.waitKey()
        red_ellipse = workingfindEllipse(red_board)
        if red_ellipse is None:
            print("No red ellipse found.")
            return None
        red_center, red_axes, red_angle = red_ellipse
        print("red ellipse = ", red_center, red_axes, red_angle)
        red_bbox = getSquareBboxForEllipse(red_center, red_axes, red_board.shape[0], red_board.shape[1])
        resized_red_bbox = (int(red_bbox[0][0]/scale_factor_red), int(red_bbox[0][1]/scale_factor_red)), \
                   (int(red_bbox[1][0]/scale_factor_red), int(red_bbox[1][1]/scale_factor_red))
        print("red bbox = ", resized_red_bbox)
        cv2.ellipse(img, red_ellipse, (0, 0, 255), 10)
        cv2.rectangle(img, resized_red_bbox[0], resized_red_bbox[1], (0, 0, 255), 10)
        cv2.imshow("red ellipse", img)
        cv2.waitKey()
        """ #for c in contours_cropped:
            #cv2.drawContours(img[resized_bbox[0][1]:resized_bbox[1][1], resized_bbox[0][0]:resized_bbox[1][0]], [c], -1, (0, 255, 0), 10)
            #cv2.imshow("contours", img[resized_bbox[0][1]:resized_bbox[1][1], resized_bbox[0][0]:resized_bbox[1][0]])
            #cv2.waitKey()
        print("board center = ", board_center, radius)
        #remapped_center = (int(board_center[0]/scale_factor_circle + resized_bbox[0][0]), int(board_center[1]/scale_factor_circle + resized_bbox[0][1]))
        remapped_center = (int(board_center[0]/scale_factor_circle + resized_bbox[0][0]+bullseye_bbox), int(board_center[1]/scale_factor_circle + resized_bbox[0][1]+bullseye_bbox))
        print("remapped center = ", (int(remapped_center[0]), int(remapped_center[1])))    
        cv2.circle(img, remapped_center, 10, (255, 0, 255), -1)
        demo_resize_center = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)
        cv2.imshow("circle", demo_resize_center)
        cv2.waitKey() """

    reformatted_bbox = [resized_bbox[0][1], resized_bbox[1][1], 
                       resized_bbox[0][0], resized_bbox[1][0]]
    
    #these lines are for visualization purposes only
    #resized_ellipse = (ellipseCenter[0]/scale_factor, ellipseCenter[1]/scale_factor), (axes[0]/scale_factor, axes[1]/scale_factor), angle
    

    cv2.circle(img, rescaled_ellipseCenter, radius, (0, 0, 255), 10)  
    cv2.imshow("circle", img)
    cv2.waitKey()

    rescaled_axes = (int(axes[0]/scale_factor_ellipse), int(axes[1]/scale_factor_ellipse))
    print("rotated rectangle = ", ellipseCenter, axes, angle)
    top_left = (int((ellipseCenter[0] - axes[0]/2)/scale_factor_ellipse), int((ellipseCenter[1] - np.sin(180-angle)*axes[0]/2)/scale_factor_ellipse))
    bottom_right = (int((ellipseCenter[0] + axes[0]/2)/scale_factor_ellipse), int((ellipseCenter[1] + axes[1]/2)/scale_factor_ellipse))
    cv2.circle(img, top_left, 5, (0, 255, 0), -1)
    cv2.putText(img, "top left", top_left, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(img, bottom_right, 5, (0, 255, 0), -1)
    cv2.putText(img, "bottom right", bottom_right, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    rotatedRect = cv2.RotatedRect(rescaled_ellipseCenter, rescaled_axes, angle)
    vertices = cv2.boxPoints(rotatedRect)
    vertices = np.int0(vertices)
    for i in range(4):
        cv2.line(img, tuple(vertices[i]), tuple(vertices[(i+1)%4]), (255, 255, 0), 10)
    #brect = cv2.boundingRect(vertices)
    #cv2.rectangle(img, (brect[0], brect[1]), (brect[0]+brect[2], brect[1]+brect[3]), (255, 255, 0), 10) # draw the rotated rectangle
    cv2.circle(img, (int(ellipseCenter[0]/scale_factor_ellipse), int(ellipseCenter[1]/scale_factor_ellipse)), 10, (0, 255, 0), -1) #draw the center
    cv2.rectangle(img, resized_bbox[0], resized_bbox[1], (0, 255, 0), 10) #draw the bounding box
    cv2.ellipse(img, (rescaled_ellipseCenter, rescaled_axes, angle), (255, 0, 255), 10) #draw the ellipse
    demo_resize = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    cv2.imshow("Bounding box preview:", demo_resize)
    cv2.waitKey()

    return reformatted_bbox


def rotate_image(image, rotation_center, angle):
    image_center = rotation_center
    if rotation_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2) 
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

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

def workingfindEllipse(img, original_img=None, scale_factor=1):
    """
    input: binary image
    output: params of roated rectangle around ellipse
    center point, axes (long and short), angle
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
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

def findEllipse(img, img_name=None, original_img=None, scale_factor=1):
    """
    input: binary image
    output: params of roated rectangle around ellipse
    center point, axes (long and short), angle
    """
    img_size = img.shape[0]
    closed_binary = None
    cv2.imshow(f"binary with size {img_size}", img)
    cv2.waitKey()
   
    img = cv2.GaussianBlur(img.copy(), (3, 3), 0)
    cv2.imshow(f"blurred with size {img_size}", img)
    cv2.waitKey()
    
    """ opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, (100, 100))
    opened = otsu_thresholding(opened)
    cv2.imshow(f"opened with size {img_size}", opened)
    cv2.waitKey()

    open_then_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    open_then_close = otsu_thresholding(open_then_close)
    cv2.imshow(f"opened then closed with size {img_size}", open_then_close)
    cv2.waitKey()
 """
    just_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    just_close = otsu_thresholding(just_close)
    cv2.imshow(f"just closed with size {img_size}", just_close)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    kernel_sizes = [(3,3), (5, 5), (7, 7), (9, 9)]
    for kernel_size in kernel_sizes:
        output_dir = 'boards/test_size_kernels'
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        write_path = osp.join(output_dir, f"{img_size}px_kernel{kernel_size[0]}_"+osp.basename(img_name))
        closed_binary = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)) # close the gaps
        closed_binary = otsu_thresholding(closed_binary)
        try: 
            success = True
            success = cv2.imwrite(write_path, closed_binary)
            if not success:
                print(f"Failed to write image to {write_path}")
            else:
                print(f"Image written to {write_path}")
        except:
            print("could not write closed binary image to ", write_path)
        
    contours, _ = cv2.findContours(just_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    best_ellipse = None
    print("contours = ", len(contours))

    for c in contours:
        if c.shape[0] > 5:
            rectangle = cv2.fitEllipse(c)
            (x, y), (MA, ma), angle = rectangle
            size = MA * ma
            circularity = MA / ma
            ellipses.append((rectangle, size, circularity))

    if len(ellipses) == 0:
        return None
    else:
        ellipses.sort(key=lambda x: x[1], reverse=True)
    size_of_largest_ellipse = ellipses[0][1]
    print("size of largest ellipse = ", size_of_largest_ellipse)

    for i, ellipse in enumerate(ellipses):
        if ellipse[2] > 0.8:
            print("ellipse i size: ", ellipse[1])
            if ellipse[1] > 0.5 * size_of_largest_ellipse:
                best_ellipse = ellipse[0] # for now its just the largest ellipse with circularity > 0.8 and it must be at least 50% of the largest ellipse
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


def find_boardCenterMinEnclosing(crop_img):

    img_opened = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.imshow("opened", img_opened)
    cv2.waitKey()
    crop_img_closed_circle = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv2.imshow("closed", crop_img_closed_circle)
    cv2.waitKey()
    contours, hierarchy = cv2.findContours(img_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("contours = ", len(contours))
    nr_of_levels = len(hierarchy[0])
    print("nr of levels = ", nr_of_levels)
    best_circle = cv2.minEnclosingCircle(contours[len(contours)-1])
    """ for i, c in enumerate(contours):
        # take contour with no child
        if hierarchy[0][i][2] == -1:
            best_cirlce = cv2.minEnclosingCircle(c)
        if c.shape[0] > 5:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if best_cirlce is None or radius < best_cirlce[1]:
                best_cirlce = ((x, y), radius)
 """
    cv2.circle(img_opened, (int(best_circle[0][0]), int(best_circle[0][1])), int(best_circle[1]), (0, 255, 0), 10)
    #cv2.imshow("circle", img_opened)
    cv2.waitKey()

    return contours, best_circle[0], int(best_circle[1]) 

def find_boardCenterHough(crop_img):
    cv2.imshow("Houghcircles", crop_img)
    circles = cv2.HoughCircles(crop_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    print("Found circles = ", circles)
    if circles is None:
        return None, None, None
    best_circle = circles[0][0]
    circles = np.uint16(np.around(circles))
    """   for i in circles[0, :]:
        cv2.circle(crop_img, (i[0], i[1]), i[2], (0, 255, 0), 10)
        cv2.circle(crop_img, (i[0], i[1]), 2, (0, 0, 255), 3) """
    cv2.circle(crop_img, (int(best_circle[0]), int(best_circle[1])), int(best_circle[2]), (0, 255, 0), 10)
    cv2.imshow("Houghcircles", crop_img)
    cv2.waitKey()
    return best_circle


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Board detection using ellipse fitting.')
    parser.add_argument('-i', '--input', help='Path to input image.', 
                       default='boards/myboard/testboard.jpg')
    args = parser.parse_args()
    
    if osp.isdir(args.input):
        for img in os.listdir(args.input):
            if not osp.isdir(osp.join(args.input, img)):
                print("bbox = ", find_board_vEllipse2(osp.join(args.input, img)))
    else:
        print("bbox = ", find_board_vEllipse2(args.input))