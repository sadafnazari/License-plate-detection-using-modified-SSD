
from __future__ import division

import numpy as np
import parameters as para
from onehotcode import onehotencode
import math

anchors = np.array(para.ANCHORS, dtype=np.float32)


def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits))


def softmax_logits(labels, logits):
    sum_ = 0.0
    # print('len labels: ', len(labels), 'len logits: ', len(logits))
    for i in range(len(labels)):
        sum_ = labels[i] * (-1) * math.log(softmax(logits[i]))
    return sum_


def calculate_iou(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    intersection = w * h
    union = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - intersection

    if union <= 0.0:
        iou = 0.0
    else:
        iou = 1.0 * intersection / union
    return iou


def smooth_l1(x):
    # print('smooth l1:', x)
    if abs(x) < 1:
        return 0.5*(x**2)
    else:
        return abs(x)-0.5


def smooth_l1_loss(pred, anchor_cx, anchor_cy, anchor_w, anchor_h, gt):

    g_cx1 = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
    g_cy1 = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
    g_w1 = math.log(gt[2]/anchor_w)*para.W_SCALE
    g_h1 = math.log(gt[3]/anchor_h)*para.H_SCALE

    g_cx2 = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
    g_cy2 = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
    g_w2 = math.log(gt[2]/anchor_w)*para.W_SCALE
    g_h2 = math.log(gt[3]/anchor_h)*para.H_SCALE

    pred_cx1, pred_cy1, pred_w1, pred_h1 = pred[0], pred[1], pred[2], pred[3]
    pred_cx2, pred_cy2, pred_w2, pred_h2 = pred[4], pred[5], pred[6], pred[7]


    loss = smooth_l1(pred_cx1-g_cx1) + smooth_l1(pred_cy1-g_cy1) + smooth_l1(pred_w1-g_w1) + smooth_l1(pred_h1-g_h1)
    loss += smooth_l1(pred_cx2-g_cx2) + smooth_l1(pred_cy2-g_cy2) + smooth_l1(pred_w2-g_w2) + smooth_l1(pred_h2-g_h2)


    return loss



def merge_loss(pos_loss, conf_loss, num_pos):
    num_neg = int(para.RATIO*num_pos)
    temp = np.sort(conf_loss)
    top_k_loss = temp[-1*num_neg:]
    neg_loss = 0.0
    for i in top_k_loss:
        neg_loss += i
    loss = (pos_loss+neg_loss)/num_pos
    return loss


def validation_loss(grondtruth, loc, cls):
    # print('gt: ', grondtruth)
    # print('loc: ', loc[0])
    # print('cls: ', cls[0])
    gt_cls_bg = np.squeeze(onehotencode([para.LABELS.Class_name[para.NUM_CLASSESS-1]+'_*']))
    conf_loss = []
    pos_loss = 0.0
    counter = 0
    for i in anchors:
        pos_l = 0.0
        anchor_cx, anchor_cy, anchor_w, anchor_h = i[0], i[1], i[2], i[3]
        anchor_xmin = anchor_cx - anchor_w * 0.5
        anchor_ymin = anchor_cy - anchor_h * 0.5
        anchor_xmax = anchor_cx + anchor_w * 0.5
        anchor_ymax = anchor_cy + anchor_h * 0.5
        for j in range(para.MAX_NUM_GT):
            gt_loc = grondtruth[0:8]
            gt_cls = grondtruth[8::]

            # xmin
            gt_x1 = gt_loc[0] - 0.5 * gt_loc[2]  # checked
            gt_x4 = gt_loc[4] - 0.5 * gt_loc[6]  # checked

            # xmax
            gt_x3 = gt_loc[0] + 0.5 * gt_loc[2]  # checked
            gt_x2 = gt_loc[4] + 0.5 * gt_loc[6]  # checked

            # ymin
            gt_y1 = gt_loc[1] - 0.5 * gt_loc[3]  # checked
            gt_y2 = gt_loc[5] - 0.5 * gt_loc[7]  # checked

            # ymax
            gt_y3 = gt_loc[1] + 0.5 * gt_loc[3]  # checked
            gt_y4 = gt_loc[5] + 0.5 * gt_loc[7]  # checked

            gt_xmin = min(gt_x1, gt_x4)
            gt_ymin = min(gt_y1, gt_y2)
            gt_xmax = max(gt_x2, gt_x3)
            gt_ymax = max(gt_y3, gt_y4)

            iou = calculate_iou(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax)

            loc_l = smooth_l1_loss(loc[counter], anchor_cx, anchor_cy, anchor_w, anchor_h, gt_loc)  # loc loss(positive)
            conf_l = softmax_logits(labels=gt_cls, logits=cls[counter])  # conf loss(positive)
            if iou > para.MATCH_IOU:
                mask = 1
            else:
                mask = 0
            pos_l = pos_l + mask * (conf_l + para.ALPHA * loc_l)


        if pos_l == 0.0:
            flag = 1
        else:
            flag = 0
        loss = flag * softmax_logits(labels=gt_cls_bg, logits=cls[counter])  # conf loss(neg)
        conf_loss.append(loss)
        pos_loss = pos_loss + pos_l


    non_zero = 0
    for i in conf_loss:
        if i != 0.0:
            non_zero += 1

    num_pos = len(anchors) - non_zero

    if num_pos == 0:
        return 0
    else:
        return merge_loss(pos_loss,conf_loss,num_pos)

