'''
Cell Motion Field Generator
'''

import cv2
import numpy as np
import os
import argparse


def compute_vector(black, pre, nxt, result, result_y, result_x, sgm):
    v = nxt - pre
    if (np.linalg.norm(v) != 0):
        v = v / np.linalg.norm(v)

    up = np.array([-v[1], v[0]]) * sgm
    dw = np.array([v[1], -v[0]]) * sgm

    v1 = up + nxt
    v2 = dw + nxt
    v3 = up + pre
    v4 = dw + pre

    points = np.round(np.array([[v1[0], v1[1]], [v2[0], v2[1]], [v4[0], v4[1]], [v3[0], v3[1]]]))

    img_t = black.copy()
    img_y = black.copy()
    img_x = black.copy()

    img_t = cv2.fillPoly(img=img_t, pts=np.int32([points]), color=1)
    # img_t = cv2.circle(img_t, (pre[0], pre[1]), sgm, (1), thickness=-1, lineType=cv2.LINE_4)
    # img_t = cv2.circle(img_t, (nxt[0], nxt[1]), sgm, (1), thickness=-1, lineType=cv2.LINE_4)

    img_y[img_t != 0] = v[1]
    img_x[img_t != 0] = v[0]
    result = result + img_t
    result_x = result_x + img_x
    result_y = result_y + img_y

    return result, result_y, result_x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('tracklet', help='path of tracklet')
    parser.add_argument('target_image', help='path of one sample image of the target')
    parser.add_argument('save_path', help='save path for output(s)')
    parser.add_argument('--sgm', type=int, default=5)
    tp = lambda x: list(map(int, x.split(',')))
    parser.add_argument('--intervals', type=tp, default=[1], help='frame intervals, please split with commas')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    sgm = args.sgm
    track_let = np.loadtxt(args.tracklet).astype('int')  # 5 columns [frame, id, x, y, parent_id]
    image_size = cv2.imread(args.target_image).shape
    frames = np.unique(track_let[:, 0])
    ids = np.unique(track_let[:, 1])
    itvs = args.intervals

    for itv in itvs:
        zeros = np.zeros((image_size[0] + sgm * 2, image_size[1] + sgm * 2, 1))
        output = []

        par_id = -1  # parent id
        for idx, i in enumerate(frames[:-itv]):
            result = zeros.copy()
            result_y = zeros.copy()
            result_x = zeros.copy()

            for j in ids:
                index_check = len(track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)])
                index_chnxt = len(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)])
                if index_chnxt != 0:
                    par_id = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0, -1]
                if (index_check != 0) & (index_chnxt != 0):
                    data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == j)][0][2:-1]
                    dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]

                    result, result_y, result_x = compute_vector(zeros, data, dnxt, result, result_y, result_x, sgm)

                elif ((index_check == 0) & (index_chnxt != 0) & (par_id != -1)):
                    try:
                        data = track_let[(track_let[:, 0] == i) & (track_let[:, 1] == par_id)][0][2:-1]
                        dnxt = track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0][2:-1]
                        result, result_y, result_x = compute_vector(zeros, data, dnxt, result, result_y, result_x, sgm)
                    except IndexError:
                        print('Error: no parent')
                        print(track_let[(track_let[:, 0] == i + itv) & (track_let[:, 1] == j)][0])

            result = result[sgm:-sgm, sgm:-sgm]
            print(i + 1, 'to', i + itv + 1, result.max())

            result_org = result.copy()
            result[result == 0] = 1
            result_y = result_y[sgm:-sgm, sgm:-sgm]
            result_x = result_x[sgm:-sgm, sgm:-sgm]
            result_x = (result_x / result)
            result_y = (result_y / result)

            result_vector = np.concatenate((result_y, result_x), axis=-1)
            output.append(result_vector)
        output = np.array(output).astype('float16')
        save_path_vector = os.path.join(args.save_path, f'cmf_{itv:02}_{sgm:03}.npy')
        np.save(save_path_vector, output)
    print('finished')
