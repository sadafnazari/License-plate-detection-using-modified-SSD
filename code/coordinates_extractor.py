from os import walk
import parameters as para
import os
import numpy as np
from onehotcode import onehotencode
from parse import parse_size
from parse import parse_object

# path = os.path.join(para.PATH,'data','ccpd_base')

# _, _, filenames = next(walk(path))



def ground_truth_four(xml):
    size_dict = parse_size(xml)

    gt = np.zeros([para.MAX_NUM_GT, 4 + para.NUM_CLASSESS], dtype=np.float32)
    object_list = parse_object(xml)
    j = 0
    for box in object_list:
        box_class = box['classes']
        x1 = box['x1'] / para.INPUT_SIZE[0]
        y1 = box['y1'] / para.INPUT_SIZE[1]
        x2 = box['x2'] / para.INPUT_SIZE[0]
        y2 = box['y2'] / para.INPUT_SIZE[1]
        x3 = box['x3'] / para.INPUT_SIZE[0]
        y3 = box['y3'] / para.INPUT_SIZE[1]
        x4 = box['x4'] / para.INPUT_SIZE[0]
        y4 = box['y4'] / para.INPUT_SIZE[1]

        # x1 = box['x1']
        # y1 = box['y1']
        # x2 = box['x2']
        # y2 = box['y2']
        # x3 = box['x3']
        # y3 = box['y3']
        # x4 = box['x4']
        # y4 = box['y4']

        # print(min(x1, x4), max(x2, x3), min(y1, y2), max(y4, y3))


        cx1 = x1 + 0.5 * (x3 - x1)
        cx2 = x4 + 0.5 * (x2 - x4)
        cy1 = y1 + 0.5 * (y3 - y1)
        cy2 = y2 + 0.5 * (y4 - y2)
        w1 = x3 - x1
        w2 = x2 - x4
        h1 = y3 - y1
        h2 = y4 - y2
        # print(w1, h1)
        #
        # print(w1/h1)
        # print(w2, h2)
        # print(w2/h2)

        gt = np.zeros([para.MAX_NUM_GT, 8 + para.NUM_CLASSESS], dtype=np.float32)
        j = 0

        # print(one, two, three, four)
        class_onehotcode = np.squeeze(onehotencode([box_class + '_*']))
        # box = np.hstack((np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32), class_onehotcode))
        box = np.hstack((np.array([cx1, cy1, w1, h1, cx2, cy2, w2, h2], dtype=np.float32), class_onehotcode))

        gt[j, :] = box
        j = j + 1
    for i in range(j, para.MAX_NUM_GT):
        if i == para.MAX_NUM_GT: break
        gt[i, :] = box

    # print({'groundtruth': gt})
    return {'groundtruth': gt}


# # print(filenames[100])
# print(path + filenames[10])
# read_dir = os.path.join(para.PATH,'data','ccpd_base',filenames[100])
# ground_truth_four(read_dir, filenames[100])
# # for i in filenames:
# #     print(ground_truth_four(filenames[0]))



###################################################################################################



# def calculate_iou(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
#     w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
#     h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
#     intersection = w * h
#     union = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - intersection
#     if union <= 0.0:
#         iou = 0.0
#     else:
#         iou = 1.0 * intersection / union
#     return iou
#
# def smooth_l1(x):
#     if abs(x) < 1:
#         return 0.5*(x**2)
#     else:
#         return abs(x)-0.5
#
#
# def smooth_l1_loss( anchor_cx, anchor_cy, anchor_w, anchor_h, gt):
#
#
#     g_cx1 = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
#     g_cy1 = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
#     g_w1 = math.log(gt[2]/anchor_w)*para.W_SCALE
#     g_h1 = math.log(gt[3]/anchor_h)*para.H_SCALE
#
#     g_cx2 = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
#     g_cy2 = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
#     g_w2 = math.log(gt[2]/anchor_w)*para.W_SCALE
#     g_h2 = math.log(gt[3]/anchor_h)*para.H_SCALE
#
#
#
#
#
#     return g_cx1, g_cy1, g_w1, g_h1, g_cx2, g_cy2, g_w2, g_h2
#
#
# path = os.path.join(para.PATH,'data','annotation','xml')
# _, _, filenames = next(walk(path))
#
# for i in filenames:
#     ground_truth_four(os.path.join(path, i))
#
# import math
# anchors = np.array(para.ANCHORS, dtype=np.float32)
# c=0
# for i in filenames:
#     print(i)
#     gt = ground_truth_four(os.path.join(path, i))
#     gt_loc = gt['groundtruth'][0]
#     # xmin
#     gt_x1 = gt_loc[0] - 0.5 * gt_loc[2]
#     gt_x4 = gt_loc[4] - 0.5 * gt_loc[6]
#
#     # xmax
#     gt_x3 = gt_loc[0] + 0.5 * gt_loc[2]
#     gt_x2 = gt_loc[4] + 0.5 * gt_loc[6]
#
#     # ymin
#     gt_y1 = gt_loc[1] - 0.5 * gt_loc[3]
#     gt_y2 = gt_loc[5] - 0.5 * gt_loc[7]
#
#     # ymax
#     gt_y3 = gt_loc[1] + 0.5 * gt_loc[3]
#     gt_y4 = gt_loc[5] + 0.5 * gt_loc[7]
#
#     gt_xmin = min(gt_x1, gt_x4)
#     gt_ymin = min(gt_y1, gt_y2)
#     gt_xmax = max(gt_x2, gt_x3)
#     gt_ymax = max(gt_y3, gt_y4)
#
#
#
#     immmm = i[0:-4]
#     path2 = os.path.join(para.PATH, 'data', 'annotation', 'images', immmm + '.jpg')
#     # print(path2)
#     import cv2 as cv
#
#     im = cv.imread(path2).astype(np.float32)
#     flag =True
#     for j in range(len(anchors)):
#         anchor = anchors[j, :]
#         anchor_cx, anchor_cy, anchor_w, anchor_h = anchor[0], anchor[1], anchor[2], anchor[3]
#         anchor_xmin = anchor_cx - anchor_w * 0.5
#         anchor_ymin = anchor_cy - anchor_h * 0.5
#         anchor_xmax = anchor_cx + anchor_w * 0.5
#         anchor_ymax = anchor_cy + anchor_h * 0.5
#
#         iou1 = calculate_iou(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax)
#         # iou2 = calculate_iou(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax, gt_xmin1, gt_ymin1, gt_xmax1, gt_ymax1)
#
#         # print(iou)
#
#         # if iou1 > 0.5:
#         #     print(iou1)
#
#         # if iou2> 0.5:
#         #     print('*', iou2)
#         if iou1 > 0.5:
#             print(iou1)
#             flag = False
#             break
#     if flag:
#         c+=1
# print(c)
            # anchor_xmin *= 300
            # anchor_ymin *= 300
            # anchor_xmax *= 300
            # anchor_ymax *= 300
            # anchor_xmin = int(anchor_xmin)
            # anchor_ymin = int(anchor_ymin)
            # anchor_xmax = int(anchor_xmax)
            # anchor_ymax = int(anchor_ymax)
            # # immmm = i[0:-4]
            # # path2 = os.path.join(para.PATH, 'data', 'annotation', 'images', immmm+'.jpg')
            # # print(path2)
            # # import cv2 as cv
            # # im = cv.imread(path2).astype(np.float32)
            # im = cv.line(im, (anchor_xmin, anchor_ymin), (anchor_xmax, anchor_ymin), (0, 255, 0), 1)
            # im = cv.line(im, (anchor_xmax, anchor_ymin), (anchor_xmax, anchor_ymax), (0, 255, 0), 1)
            # im = cv.line(im, (anchor_xmax, anchor_ymax), (anchor_xmin, anchor_ymax), (0, 255, 0), 1)
            # im = cv.line(im, (anchor_xmin, anchor_ymax), (anchor_xmin, anchor_ymin), (0, 255, 0), 1)
            # # dst = os.path.join(para.PATH, 'data', 'annotation', 'images', 'test'+immmm + '.jpg')
            # # cv.imwrite(dst, im)
            # _, _, filenames2 = next(walk(path))

            # anchor_xmin = int(anchor_xmin)
            # anchor_ymin = int(anchor_ymin)
            # anchor_xmax = int(anchor_xmax)
            # anchor_ymax = int(anchor_ymax)
            # if j< 3610:
            #     anchor_xmin *= 19
            #     anchor_ymin *= 19
            #     anchor_xmax *= 19
            #     anchor_ymax *= 19
            # elif j<4610:
            #     anchor_xmin *= 10
            #     anchor_ymin *= 10
            #     anchor_xmax *= 10
            #     anchor_ymax *= 10
            # elif j<4866:
            #     anchor_xmin *= 5
            #     anchor_ymin *= 5
            #     anchor_xmax *= 5
            #     anchor_ymax *= 5
            # elif j< 4950:
            #     anchor_xmin *= 3
            #     anchor_ymin *= 3
            #     anchor_xmax *= 3
            #     anchor_ymax *= 3
            # elif j<4990:
            #     anchor_xmin *= 2
            #     anchor_ymin *= 2
            #     anchor_xmax *= 2
            #     anchor_ymax *= 2
            # elif j<4999:
            #     anchor_xmin *= 1
            #     anchor_ymin *= 1
            #     anchor_xmax *= 1
            #     anchor_ymax *= 1
            # print(iou2)
#             # print(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)
#             # print(smooth_l1_loss(anchor_cx,anchor_cy,anchor_w,anchor_h, gt_loc))
#
#     dst = os.path.join(para.PATH, 'data', 'annotation', 'images', 'test' + immmm + '.jpg')
#     cv.imwrite(dst, im)

#####################################################################################################################



# print(ground_truth_four(pathh))

    # splited = filename.split('-')
    # coordinates = splited[3].split('_')
    # # print(coordinates)
    # three, four, one, two = coordinates[0].split('&'), coordinates[1].split('&'), coordinates[2].split('&'), coordinates[3].split('&')
    # x1, y1 = int(one[0])/w, int(one[1])/h
    # x2, y2 = int(two[0])/w, int(two[1])/h
    # x3, y3 = int(three[0])/w, int(three[1])/h
    # x4, y4 = int(four[0])/w, int(four[1])/h




