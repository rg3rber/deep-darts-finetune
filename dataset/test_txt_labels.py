import argparse
import cv2
import os

def draw_labels(path_to_labels, scale=0.5):
    nr_of_files = 0

    for file in os.listdir(path_to_labels):
        if nr_of_files == 10:
            break
        key = cv2.waitKey(100) & 0xFF 
        if key == ord('q'):
            print('Quitting...')
            cv2.destroyAllWindows()
            break
        if file.endswith('.txt'):
            label_dir = os.path.split(path_to_labels)[0]
            filename = os.path.splitext(file)[0]
            split_filename = filename.split('_')[:-2]

            reconstructed_filename = '_'.join(split_filename)

            img_path = os.path.join('cropped_images', '800', 'd3_' + reconstructed_filename, file.replace('.txt', '.jpg'))
            img = cv2.imread(img_path)
            if img is None:
                print('Image not found or cannot be read:', img_path)
                continue
            img_size =  img.shape[0]
            nr_of_dart = 1
            with open(os.path.join(path_to_labels, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    x, y, w, h = list(map(float, line[1:]))
                    print(x, y, w, h)
                    x, y, w, h = int(x * img_size), int(y * img_size), int(w * img_size), int(h * img_size)
                    print(x, y, w, h)
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                    if line[0] == '4':
                        cv2.putText(img, f"Dart {nr_of_dart}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        nr_of_dart += 1
                    else:
                        cv2.putText(img, line[0], (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3, cv2.LINE_AA)
            #img_resize = cv2.resize(img, None, fx=scale, fy=scale)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            nr_of_files += 1


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--labels-folder', type=str, default='images/labels')
    parser.add_argument('-s', '--scale', type=float, default=0.5)

    args = parser.parse_args()

    draw_labels(args.labels_folder, args.scale)
