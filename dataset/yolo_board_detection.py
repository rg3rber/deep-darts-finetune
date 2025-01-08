from ultralytics import YOLO
import argparse
import os
import os.path as osp
import cv2
import numpy as np

def predict(img_path, model_path='best.pt'):
    """
    Run YOLO prediction on a single image or directory of images
    
    Args:
        img_path (str): Path to image or directory of images
        model_path (str): Path to YOLO model weights
        
    Returns:
        results: YOLO prediction results
    """
    model = YOLO(model_path)
    results = model(img_path)
    return results

def process_predictions(results, conf_threshold=0.0):
    """
    Process prediction results and return only the highest confidence prediction
    
    Args:
        results: YOLO prediction results
        conf_threshold (float): Minimum confidence threshold (0.0 to 1.0)
        
    Returns:
        dict: Highest confidence prediction including box, confidence score, and class label
        None: If no predictions meet the confidence threshold
    """
    all_predictions = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf)
            if confidence >= conf_threshold:
                pred = {
                    'bbox': box.xyxy[0].tolist(),  # Convert tensor to list
                    'confidence': confidence,
                    'class': int(box.cls),
                }
                all_predictions.append(pred)
    
    # If no predictions meet the threshold, return None
    if not all_predictions:
        return None
    
    # Return the prediction with highest confidence
    best_prediction = max(all_predictions, key=lambda x: x['confidence'])
    return best_prediction

def draw_prediction(image_path, prediction):
    """
    Draw bounding box on image for the given prediction
    
    Args:
        image_path (str): Path to the original image
        prediction (dict): Prediction dictionary containing bbox and confidence
        
    Returns:
        tuple: (drawn_image, output_path)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get bbox coordinates and convert to integers
    x1, y1, x2, y2 = map(int, prediction['bbox'])
    
    # Draw rectangle
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Add confidence score text
    conf_text = f"Conf: {prediction['confidence']:.2f}"
    cv2.putText(img, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, color, thickness)
    
    # Create output path
    image_dir = osp.dirname(image_path)
    detected_dir = osp.join(image_dir, 'detected')
    os.makedirs(detected_dir, exist_ok=True)
    
    # Create output path in the detected directory
    output_path = osp.join(detected_dir, osp.basename(image_path))

    return img, output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO object detection.')
    parser.add_argument('-i', '--input', help='Path to input image or directory.',
                       required=True)
    parser.add_argument('-m', '--model', help='Path to model weights.',
                       default='best.pt')
    parser.add_argument('-c', '--confidence', help='Minimum confidence threshold (0.0 to 1.0)',
                       type=float, default=0.25)
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window (only save output)')
    
    args = parser.parse_args()
    
    # Check if input is directory or single image
    if osp.isdir(args.input):
        # Process all images in directory
        for img in os.listdir(args.input):
            img_path = osp.join(args.input, img)
            if osp.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"\nProcessing {img}:")
                results = predict(img_path, args.model)
                best_pred = process_predictions(results, args.confidence)
                
                if best_pred:
                    print(f"Best prediction (confidence: {best_pred['confidence']:.2f}):", best_pred)
                    drawn_img, output_path = draw_prediction(img_path, best_pred)
                    cv2.imwrite(output_path, drawn_img)
                    print(f"Saved detection to: {output_path}")
                    
                    if not args.no_display:
                        resize_img = cv2.resize(drawn_img, (800, 600))
                        cv2.imshow(f"Detection - {img}", drawn_img)
                        cv2.waitKey(0)
                else:
                    print("No predictions above confidence threshold.")
    else:
        # Process single image
        results = predict(args.input, args.model)
        best_pred = process_predictions(results, args.confidence)
        
        if best_pred:
            print(f"Best prediction (confidence: {best_pred['confidence']:.2f}):", best_pred)
            drawn_img, output_path = draw_prediction(args.input, best_pred)
            cv2.imwrite(output_path, drawn_img)
            print(f"Saved detection to: {output_path}")
            
            if not args.no_display:
                cv2.imshow("Detection", drawn_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("No predictions above confidence threshold.")