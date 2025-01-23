import argparse
from yacs.config import CfgNode as CN
import os.path as osp
import os
import cv2
import numpy as np
from time import time
from dataset.annotate import draw, get_dart_scores
from dataset.find_board import compute_cal_pts, find_board_vEllipse2
import pickle
import dataset.distort as distort
import pandas as pd


def bboxes_to_xy(bboxes, max_darts=3):
    xy = np.zeros((4 + max_darts, 3), dtype=np.float32)
    for cls in range(5):
        if cls == 0:
            dart_xys = bboxes[bboxes[:, 4] == 0, :2][:max_darts]
            xy[4:4 + len(dart_xys), :2] = dart_xys
        else:
            cal = bboxes[bboxes[:, 4] == cls, :2]
            if len(cal):
                xy[cls - 1, :2] = cal[0]
    xy[(xy[:, 0] > 0) & (xy[:, 1] > 0), -1] = 1
    if np.sum(xy[:4, -1]) == 4:
        return xy, None
    else:
        estimated = np.where(xy[:4, -1] == 0)[0]
        xy = est_cal_pts(xy)
    return xy, estimated


def est_cal_pts(xy):
    missing_idx = np.where(xy[:4, -1] == 0)[0]
    if len(missing_idx) == 1:
        if missing_idx[0] <= 1:
            center = np.mean(xy[2:4, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 0:
                xy[0, 0] = -xy[1, 0]
                xy[0, 1] = -xy[1, 1]
                xy[0, 2] = 1
            else:
                xy[1, 0] = -xy[0, 0]
                xy[1, 1] = -xy[0, 1]
                xy[1, 2] = 1
            xy[:, :2] += center
        else:
            center = np.mean(xy[:2, :2], axis=0)
            xy[:, :2] -= center
            if missing_idx[0] == 2:
                xy[2, 0] = -xy[3, 0]
                xy[2, 1] = -xy[3, 1]
                xy[2, 2] = 1
            else:
                xy[3, 0] = -xy[2, 0]
                xy[3, 1] = -xy[2, 1]
                xy[3, 2] = 1
            xy[:, :2] += center
    else:
        # TODO: if len(missing_idx) > 1
        print('Missed more than 1 calibration point')
    return xy


def predictTest(
        yolo,
        cfg,
        img_folder="myTest_PXL_bbox_crop",
        labels_path='./dataset/labels.pkl',
        dataset='myTest',
        split='test',
        max_darts=3,
        write=False):

    np.random.seed(0)


    write_dir = osp.join('./models', cfg.model.name, 'preds', split)
    if write:
        os.makedirs(write_dir, exist_ok=True)

    calibrationErrorFile = pd.DataFrame(columns = ["img_name", "gt", "pred", "total error", "mean error", "inferred cal pt"])

    #data = get_splits(labels_path, dataset, split)  
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, img_folder, name) for name in os.listdir(osp.join(img_prefix, img_folder))]
    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    print('Making predictions with {}...'.format(cfg.model.name))
    allAblations = ["pure", "blurred5by5", "blurred3by3", "blurred7by7", "blurred_contrast", "higher_contrast150", "higher_contrast175"]
    #ablations = ["higher_contrast125", "lower_contrast075"]
    ablations = [""]

    # for each ablation do predictions and wirte the images to a different \dataset\ablation folder and store the results in a pickle file with its ablation name
    for ablation in ablations:
        #failedImages = []
        for i in range(len(img_paths)):
            img = cv2.imread(img_paths[i])
            if i == 1:
                ti = time()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ablation == "blurred3by3":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=3)
                img = transformimg
            elif ablation == "blurred5by5":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=5)
                img = transformimg
            elif ablation == "blurred7by7":
                imgCopy = img.copy()
                transformimg = distort.low_pass_filter(imgCopy, kernel_size=7)
                img = transformimg
            elif ablation == "blurred_contrast":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy)
                img = distort.low_pass_filter(transformimg)
            elif ablation == "higher_contrast150":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy, 1.5)
                img = transformimg
            elif ablation == "higher_contrast175":
                imgCopy = img.copy()
                transformimg = distort.change_contrast(imgCopy, 1.75)
                img = transformimg
            if i%10 == 0:
                print('Processing', i, 'of', len(img_paths))
           
            bboxes = yolo.predict(img)
            preds[i], hadToEstimateId = bboxes_to_xy(bboxes, max_darts)
            if (hadToEstimateId is not None):
                print("Had to estimate calibration point: ", hadToEstimateId, " for image: ", img_paths[i], )
            if write:
                write_dir = osp.join('.\models', cfg.model.name, 'preds', split)
                print("write dir: ", write_dir)
                os.makedirs(write_dir, exist_ok=True)
                xy = preds[i]
                xy = xy[xy[:, -1] == 1]
                gt = pd.read_pickle(labels_path)
                print("gt: ", gt)
                gt = gt[gt.img_name == osp.basename(img_paths[i])]
                gtCalibrationPts = gt.iloc[0].xy[:4]
                print("gtCalibrationPts: ", gtCalibrationPts)
                print("xy: ", xy)
                if(len(xy) < 4):
                    calibrationErrorFile = calibrationErrorFile.append({
                        "img_name": osp.basename(img_paths[i]),
                        "gt": gtCalibrationPts,
                        "pred": xy,
                        "total error": "N/A",
                        "mean error": "N/A",
                        "inferred cal pt": hadToEstimateId if hadToEstimateId is not None else "N/A",
                    }, ignore_index=True)
                    continue
                calErrorSum = 0
                for j in range(4):
                    calErrorSum += np.linalg.norm(gtCalibrationPts[j] - xy[j][:2])
                print("calibration error: ", calErrorSum)
                calibrationErrorFile = calibrationErrorFile.append({
                    "img_name": osp.basename(img_paths[i]),
                    "gt": gtCalibrationPts,
                    "pred": xy,
                    "total error": calErrorSum,
                    "mean error": calErrorSum / 4,
                    "inferred cal pt": hadToEstimateId if hadToEstimateId is not None else "N/A",
                }, ignore_index=True)

                if not args.fail_cases:
                    img = draw(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), xy[:, :2], cfg, circles=False, score=True)
                    print("writing to: " + osp.join(write_dir, ablation, os.path.basename(img_paths[i])))
                    os.makedirs(osp.join(write_dir, ablation), exist_ok=True)
                    cv2.imwrite(osp.join(write_dir, ablation, os.path.basename(img_paths[i])), img)

        fps = (len(img_paths) - 1) / (time() - ti)
        print('FPS: {:.2f}'.format(fps))
        calibrationErrorFile.to_csv(osp.join(write_dir, "calibrationErrorFile.csv"))
        print("wrote calibrationErrorFile to: ", osp.join(write_dir, "calibrationErrorFile.csv"))
"""
        ASE = []  # absolute score error
        for pred, gt in zip(preds, xys):
            ASE.append(abs(
                sum(get_dart_scores(pred[:, :2], cfg, numeric=True)) -
                sum(get_dart_scores(gt[:, :2], cfg, numeric=True))))

        ASE = np.array(ASE)
        PCS = len(ASE[ASE == 0]) / len(ASE) * 100
        MASE = np.mean(ASE)

        results = {
            'failed images': failedImages,
            'fps': fps,
            'ASE': ASE.tolist() if isinstance(ASE, np.ndarray) else ASE,  # Convert numpy array to list
            'PCS': float(PCS),
            'MASE': float(MASE)
        }"""

        # Save results as JSON
        #with open(osp.join('./models', cfg.model.name, ablation + '_pink_results.txt'), 'w') as f:
        #    json.dump(results, f)

def predictTestCompCalPts(
        yolo,
        cfg,
        img_folder="myTest_PXL_bbox_crop",
        labels_path='./dataset/labels.pkl',
        split='test',
        max_darts=3,
        write=False):

    np.random.seed(0)

    write_dir = osp.join('./models', cfg.model.name, 'preds', split)
    if write:
        os.makedirs(write_dir, exist_ok=True)

    calibrationErrorFile = pd.DataFrame(columns = ["img_name", "gt", "pred", "total error", "mean error", "inferred cal pt"])

    #data = get_splits(labels_path, dataset, split)  
    img_prefix = osp.join(cfg.data.path, 'cropped_images', str(cfg.model.input_size))
    img_paths = [osp.join(img_prefix, img_folder, name) for name in os.listdir(osp.join(img_prefix, img_folder))]
    preds = np.zeros((len(img_paths), 4 + max_darts, 3))
    print('Making predictions with {}...'.format(cfg.model.name))

    for i in range(len(img_paths)):
        img = cv2.imread(img_paths[i])
        bbox, preciseEllipse = find_board_vEllipse2(img_paths[i])
        comp_cal_pts = compute_cal_pts(preciseEllipse, img)
        normalizedCalPts = np.array(comp_cal_pts) / img.shape[0] # normalize the calibration points
        if write:
            write_dir = osp.join('.\models', cfg.model.name, 'preds', split)
            print("write dir: ", write_dir)
            os.makedirs(write_dir, exist_ok=True)
            xy = normalizedCalPts
            gt = pd.read_pickle(labels_path)
            gt = gt[gt.img_name == osp.basename(img_paths[i])]
            gtCalibrationPts = gt.iloc[0].xy[:4]
            if(len(xy) < 4):
                calibrationErrorFile = calibrationErrorFile.append({
                    "img_name": osp.basename(img_paths[i]),      
                    "gt": gtCalibrationPts,
                    "pred": xy,
                    "total error": "N/A",
                    "mean error": "N/A",
                    "inferred cal pt": "N/A",
                    }, ignore_index=True)
                continue
            calErrorSum = 0
            for j in range(4):
                calErrorSum += np.linalg.norm(gtCalibrationPts[j] - xy[j][:2])
            print("calibration error: ", calErrorSum)
            calibrationErrorFile = calibrationErrorFile.append({
                "img_name": osp.basename(img_paths[i]),
                "gt": gtCalibrationPts,
                "pred": xy,
                "total error": calErrorSum,
                "mean error": calErrorSum / 4,
                "inferred cal pt": "N/A",
            }, ignore_index=True)
            img = cv2.ellipse(img, (preciseEllipse.centerFloat, preciseEllipse.axesFloat, preciseEllipse.angle), (255, 0, 255), 2)
            for k, gtCalPt in enumerate(gtCalibrationPts):
                img = cv2.circle(img, (int(gtCalPt[0] * img.shape[0]), int(gtCalPt[1] * img.shape[0])), 1, (255, 0, 0), -1)
                img = cv2.putText(img, f"gt cal {k+1}", (int(gtCalPt[0] * img.shape[0])+10, int(gtCalPt[1] * img.shape[0])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 5)
            img = draw(img, xy[:, :2], cfg, circles=False, score=True)

            print("writing to: " + osp.join(write_dir, os.path.basename(img_paths[i])))
            os.makedirs(osp.join(write_dir), exist_ok=True)
            cv2.imwrite(osp.join(write_dir, os.path.basename(img_paths[i])), img)

    calibrationErrorFile.to_csv(osp.join(write_dir, "calibrationErrorFile.csv"))
    print("wrote calibrationErrorFile to: ", osp.join(write_dir, "calibrationErrorFile_board_angles.csv"))

if __name__ == '__main__':
    import sys
    sys.path.append('../../')   
    from train import build_model
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='deepdarts_d3')
    parser.add_argument('-s', '--split', default='test')
    parser.add_argument('-w', '--write', action='store_true')
    parser.add_argument('-p', '--compute_cal_pts', action='store_true')
    parser.add_argument('-i', '--image_folder', default='myTest_PXL')
    parser.add_argument('-lp', '--labels_path', default='dataset/labels.pkl')
    args = parser.parse_args()

    cfg = CN(new_allowed=True)
    cfg.merge_from_file(osp.join('configs', args.cfg + '.yaml'))
    cfg.model.name = args.cfg

    yolo = build_model(cfg)
    yolo.load_weights(osp.join('models', args.cfg, 'weights'), cfg.model.weights_type)
    if (args.compute_cal_pts):
        print('Computing cal points with: ', cfg.model.name)
        predictTestCompCalPts(yolo, cfg, img_folder=args.image_folder, labels_path=args.labels_path, split=args.split, write=args.write)
    else:
        print('Predicting with', cfg.model.name)
        predictTest(yolo, cfg, img_folder=args.image_folder, labels_path=args.labels_path, split=args.split, write=args.write)