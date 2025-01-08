import argparse
import os
import os.path as osp
import cv2
import pandas as pd
import numpy as np
from annotate import crop_board, draw, draw_circles, transform, get_dart_scores, total_score
from yacs.config import CfgNode as CN



def draw_warped(img, xy, cfg, circles=True, score=True, color=(255, 255, 0)):
    """Draw darts and scores on warped image"""
    xy = np.array(xy)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    line_type = 5
    
    if xy.shape[0] > 7:
        xy = xy.reshape((-1, 2))
        
    # Transform points and image
    xy_warped, img_warped, _ = transform(xy.copy(), img.copy(), angle=0)
    
    h, w = img_warped.shape[:2]
    if np.mean(xy_warped) < 1:
        xy_warped[:, 0] *= w
        xy_warped[:, 1] *= h
        
    if xy_warped.shape[0] >= 4 and circles:
        img_warped = draw_circles(img_warped, xy_warped, cfg)
        
    if xy_warped.shape[0] > 4 and score:
        scores = get_dart_scores(xy, cfg)  # Use original xy for scores
        cv2.putText(img_warped, str(total_score(scores)), (50, 50), font,
                    font_scale, (255, 255, 255), line_type)
                    
    for i, [x, y] in enumerate(xy_warped):
        x = int(round(x))
        y = int(round(y))
        if i < 4:
            c = (0, 255, 0)  # green for calibration points
        else:
            c = color  # color for dart points
            
        if i >= 4:
            cv2.circle(img_warped, (x, y), 10, c, 1)
            if score:
                txt = str(scores[i - 4])
            else:
                txt = str(i + 1)
            cv2.putText(img_warped, txt, (x + 8, y), font,
                    font_scale, c, line_type)
        else:
            cv2.circle(img_warped, (x, y), 10, c, 1)
            cv2.putText(img_warped, str(i + 1), (x + 8, y), font,
                        font_scale/2, c, line_type)
                        
    return img_warped

def create_side_by_side(img1, img2, img_name):
    """Create a side by side comparison image with labels"""
    # Ensure both images are the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    
    # Resize images to match height while maintaining aspect ratio
    if h1 != h:
        w1 = int(w1 * (h / h1))
        img1 = cv2.resize(img1, (w1, h))
    if h2 != h:
        w2 = int(w2 * (h / h2))
        img2 = cv2.resize(img2, (w2, h))
    
    # Create combined image
    combined = np.zeros((h + 40, w1 + w2 + 20, 3), dtype=np.uint8)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, f"Original - {img_name}", (10, 30), 
                font, 0.7, (255, 255, 255), 1)
    cv2.putText(combined, f"Warped - {img_name}", (w1 + 30, 30), 
                font, 0.7, (255, 255, 255), 1)
    
    # Copy images
    combined[40:40+h, :w1] = img1
    combined[40:40+h, w1+20:w1+20+w2] = img2
    
    return combined

def main(cfg, folder, scale, draw_circles, dart_score=True):
    img_dir = osp.join("../", cfg.data.path, 'images', folder)
    imgs = sorted(os.listdir(img_dir))
    annot_path = osp.join("../", cfg.data.path, 'annotations', folder + '.pkl')
    
    if osp.isfile(annot_path):
        annot = pd.read_pickle(annot_path)
    else:
        print('No annotations found for this folder')
        return

    # Create output directory
    output_dir = osp.join("../", cfg.data.path, 'visualizations', folder)
    os.makedirs(output_dir, exist_ok=True)
        
    for i in range(len(annot)):
        a = annot.iloc[i]
        if a['xy'] is not None and a['bbox'] is not None:
            # Load and crop image
            img_path = osp.join(img_dir, a['img_name'])
            crop, _ = crop_board(img_path, bbox=a['bbox'])
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
            
            # Draw original image
            original = draw(crop.copy(), np.array(a['xy']), cfg, draw_circles, dart_score)
            
            # Draw warped image
            warped = draw_warped(crop.copy(), np.array(a['xy']), cfg, draw_circles, dart_score)
            
            # Combine images
            combined = create_side_by_side(original, warped, a['img_name'])
            
            # Save combined image
            output_path = osp.join(output_dir, f"comparison_{a['img_name']}")
            cv2.imwrite(output_path, combined)
            print(f"Saved {output_path}")

if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--img-folder', default='d3_02_12_2024_1')
    parser.add_argument('-s', '--scale', type=float, default=0.5)
    parser.add_argument('-d', '--draw-circles', action='store_true')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file('../configs/deepdarts_d3.yaml')

    main(cfg, args.img_folder, args.scale, args.draw_circles)