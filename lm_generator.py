'''
Likelihood Map Generator
'''

import cv2
import numpy as np
import os
import argparse

def compute_vector(black, pre, nxt, result, result_l, result_y, result_x, result_z, blur_range, z_value):
    img_l = black.copy()  # likelihood image
    img_l[nxt[1] + blur_range, nxt[0] + blur_range] = 255
    img_l = cv2.GaussianBlur(img_l, ksize=(int(blur_range * 2) + 1, int(blur_range * 2) + 1), sigmaX=6)
    img_l = img_l / img_l.max()
    points = np.where(img_l > 0)
    img_y = black.copy()
    img_x = black.copy()
    img_z = black.copy()
    for y, x in zip(points[0], points[1]):
        v3d = pre + [blur_range, blur_range] - [x, y]
        v3d = np.append(v3d, z_value)
        v3d = v3d / np.linalg.norm(v3d) * img_l[y, x]
        img_y[y, x] = v3d[1]
        img_x[y, x] = v3d[0]
        img_z[y, x] = v3d[2]

    img_i = result_l - img_l
    result_y = np.where(img_i < 0, img_y, result_y)
    result_x = np.where(img_i < 0, img_x, result_x)
    result_z = np.where(img_i < 0, img_z, result_z)

    img_i = img_l.copy()
    img_i[img_i == 0] = 2
    img_i = result_l - img_i
    result[img_i == 0] += 1
    result_y += np.where(img_i == 0, img_y, 0)
    result_x += np.where(img_i == 0, img_x, 0)
    result_z += np.where(img_i == 0, img_z, 0)

    result_l = np.maximum(result_l, img_l)
    return result, result_l, result_y, result_x, result_z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tracklet', help='path of tracklet')
    parser.add_argument('target_image', help='path of one sample image of the target')
    parser.add_argument('save_path', help='save path for output(s)')
    parser.add_argument('--blur_range', type=int, default=50)
    parser.add_argument('--sgm', type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    blur_range = args.blur_range
    sgm = args.sgm
    track_let = np.loadtxt(args.tracklet).astype('int') # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(args.target_image).shape
    frames = np.unique(track_let[:, 0])

    zeros = np.zeros((image_size[0], image_size[1]))
    output = []

    for i in frames:
        print(i)
        result = zeros.copy()
        data_frames = track_let[track_let[:, 0] == i]
        for data_frame in data_frames:
            x = data_frame[2]
            y = data_frame[3]
            img_t = zeros.copy()  # likelihood map of one cell
            img_t[y][x] = 255  # plot a white dot
            img_t = np.pad(img_t, (blur_range, blur_range), 'constant')  # zero padding
            img_t = cv2.GaussianBlur(img_t, ksize=(blur_range*2+1, blur_range*2+1), sigmaX=sgm)  # filter gaussian(hyper_parameter)
            img_t = img_t[blur_range:-blur_range, blur_range:-blur_range]  # remove padding
            result = np.maximum(result, img_t)  # compare result with gaussian_img

        result = (result / result.max()) * 255
        result = result.astype('uint8')

        save_path_img = os.path.join(args.save_path, f'lm_{i:03}.png')
        cv2.imwrite(save_path_img, result)
        output.append(result)
    save_path_lm = os.path.join(args.save_path, f'lm_{blur_range:03}_{sgm:02}.npy')
    np.save(save_path_lm, output)
    print('finished')

