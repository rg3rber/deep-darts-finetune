import cv2
import numpy as np

def prepare_reference_dartboard(image_path, inner_ratio=0.8, outer_ratio=0.95):
    """
    Extract just the colored playing section of a dartboard image.
    
    Args:
        image_path: Path to the dartboard image
        inner_ratio: How much of the inner part to exclude (numbers area)
        outer_ratio: How much of the outer part to include (excludes outer ring)
    
    Returns:
        tuple: (color_version, gray_version, mask) of the processed reference
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not open image")
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    print("image read and removed noise")
    
    # Find the circle of the dartboard
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=img.shape[0]/2,
        param1=50,
        param2=30,
        minRadius=int(min(img.shape[:2])/4),
        maxRadius=int(min(img.shape[:2])/2)
    )
    
    print("circle found")
    if circles is None:
        raise ValueError("Could not detect dartboard circle")
    
    # Get the largest circle
    circle = circles[0][0]
    center = (int(circle[0]), int(circle[1]))
    radius = int(circle[2])
    
    # Create a mask for the playing section
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    outer_radius = int(radius * outer_ratio)
    inner_radius = int(radius * inner_ratio)
    
    # Draw the donut shape mask
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    
    # Apply mask to both color and grayscale versions
    color_result = cv2.bitwise_and(img, img, mask=mask)
    gray_result = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Crop to bounding box of the circle
    x1 = max(0, center[0] - outer_radius)
    y1 = max(0, center[1] - outer_radius)
    x2 = min(img.shape[1], center[0] + outer_radius)
    y2 = min(img.shape[0], center[1] + outer_radius)
    
    color_result = color_result[y1:y2, x1:x2]
    gray_result = gray_result[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]
    
    return color_result, gray_result, mask

def test_matching_method(query_path, reference_path, use_color=True):
    """
    Test matching using either color or grayscale and return matching metrics.
    """
    # Prepare reference image
    reference_board = cv2.imread(reference_path)
    color_ref, gray_ref = cv2.cvtColor(reference_board, cv2.COLOR_BGR2HSV), cv2.cvtColor(reference_board, cv2.COLOR_BGR2GRAY)
    print("reference prepared")
    #save the reference images
    cv2.imwrite('images/reference.jpg', color_ref)
    cv2.imwrite('images/reference_gray.jpg', gray_ref)
    
    # Read and resize query image to match reference size
    query = cv2.imread(query_path)

    #remove noise
    query = cv2.GaussianBlur(query, (5, 5), 0)
    gray_reference_board = cv2.GaussianBlur(gray_ref, (5, 5), 0)

    resized_ref = cv2.resize(gray_reference_board, (800, 800))
    cv2.imwrite('images/gray_ref_blur.png', resized_ref)
    

    print("wrote blurred and resized reference image")


    if query is None:
        raise ValueError("Could not open query image")
    
    # Choose color or grayscale processing
    if use_color:
        reference = cv2.cvtColor(color_ref, cv2.COLOR_BGR2HSV)
        query = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    else:
        reference = gray_ref
        query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

    print("image read. ready")
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(query, None)
    
    if des1 is None or des2 is None:
        return {
            'num_keypoints_ref': len(kp1) if kp1 else 0,
            'num_keypoints_query': len(kp2) if kp2 else 0,
            'num_good_matches': 0,
            'avg_match_distance': float('inf')
        }
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Calculate metrics
    avg_distance = np.mean([m.distance for m in good_matches]) if good_matches else float('inf')
    
    return {
        'num_keypoints_ref': len(kp1),
        'num_keypoints_query': len(kp2),
        'num_good_matches': len(good_matches),
        'avg_match_distance': avg_distance
    }

if __name__ == '__main__':
    # Test the matching method
    metrics_color = test_matching_method('images/testboard.jpg', 'images/dartboard_ref.png', use_color=True)
    metrics_gray = test_matching_method('images/testboard.jpg', 'images/dartboard_ref.png', use_color=False)
    
    print("Color matching metrics:", metrics_color)
    print("Grayscale matching metrics:", metrics_gray)