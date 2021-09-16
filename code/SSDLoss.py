# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import tensorflow as tf
import parameters as para
from onehotcode import onehotencode
import sys


anchors = np.array(para.ANCHORS, dtype=np.float32)
anchors_tensor = tf.convert_to_tensor(anchors, dtype=tf.float32)



def calculate_iou(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = tf.maximum(0.0, tf.minimum(xmax0, xmax1) - tf.maximum(xmin0, xmin1))
    h = tf.maximum(0.0, tf.minimum(ymax0, ymax1) - tf.maximum(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    iou = tf.reduce_max([0.0,intersection/union])
    return iou


def smooth_l1(x):
    l1 = (0.5*(x**2))*tf.cast(tf.less(tf.abs(x),1.0),dtype=tf.float32)
    l2 = (tf.abs(x)-0.5)*tf.cast(tf.greater_equal(tf.abs(x),1.0),dtype=tf.float32)
    return l1+l2


def smooth_l1_loss(pred, anchor_cx, anchor_cy, anchor_w, anchor_h, gt):

    g_cx1 = ((gt[0]-anchor_cx)/anchor_w)*para.X_SCALE
    g_cy1 = ((gt[1]-anchor_cy)/anchor_h)*para.Y_SCALE
    g_w1 = tf.log(gt[2]/anchor_w)*para.W_SCALE
    g_h1 = tf.log(gt[3]/anchor_h)*para.H_SCALE

    g_cx2 = ((gt[4]-anchor_cx)/anchor_w)*para.X_SCALE
    g_cy2 = ((gt[5]-anchor_cy)/anchor_h)*para.Y_SCALE
    g_w2 = tf.log(gt[6]/anchor_w)*para.W_SCALE
    g_h2 = tf.log(gt[7]/anchor_h)*para.H_SCALE

    pred_cx1, pred_cy1, pred_w1, pred_h1 = pred[0], pred[1], pred[2], pred[3]
    pred_cx2, pred_cy2, pred_w2, pred_h2 = pred[4], pred[5], pred[6], pred[7]


    loss = smooth_l1(pred_cx1-g_cx1) + smooth_l1(pred_cy1-g_cy1) + smooth_l1(pred_w1-g_w1) + smooth_l1(pred_h1-g_h1)
    loss += smooth_l1(pred_cx2-g_cx2) + smooth_l1(pred_cy2-g_cy2) + smooth_l1(pred_w2-g_w2) + smooth_l1(pred_h2-g_h2)


    return loss

    
def merge_loss(pos_loss, conf_loss, num_pos):
    num_neg = tf.cast(para.RATIO*num_pos, tf.int32)
    top_k_loss, _ = tf.nn.top_k(conf_loss, k=tf.minimum(num_neg, len(anchors)))
    neg_loss = tf.reduce_sum(top_k_loss)
    loss = (pos_loss+neg_loss)/num_pos
    return loss


def ssd_loss(loc, cls, grondtruth):
    losses = []
    noumber_of_positives = []
    positive_loss = []
    num_anchors = len(anchors)
    gt_cls_bg = np.squeeze(onehotencode([para.LABELS.Class_name[para.NUM_CLASSESS-1]+'_*']))
    for b in range(para.BATCH_SIZE):
        i = 0
        m = 0
        pos_loss1 = 0.0
        pos_loss2 = 0.0
        loc_b = loc[b, :, :]
        cls_b = cls[b, :, :]
        grondtruth_b = grondtruth[b, :, :]
        
        def cond1(i,pos_loss1,conf_loss1):
            boolean = tf.less(i, num_anchors)
            return boolean

        def body1(i, pos_loss1, conf_loss1):
            pos_l = 0.0
            pred_loc = loc_b[i,:]
            pred_conf = cls_b[i,:]
            anchor = anchors_tensor[i,:]
            
            anchor_cx,anchor_cy,anchor_w,anchor_h = anchor[0], anchor[1], anchor[2], anchor[3]
            anchor_xmin = anchor_cx - anchor_w*0.5
            anchor_ymin = anchor_cy - anchor_h*0.5
            anchor_xmax = anchor_cx + anchor_w*0.5
            anchor_ymax = anchor_cy + anchor_h*0.5


            for j in range(para.MAX_NUM_GT):
                gt_loc = grondtruth_b[j,0:8]
                gt_cls = grondtruth_b[j,8::]


                gt_xmin1 = gt_loc[0] - 0.5 * gt_loc[2]
                gt_ymin1 = gt_loc[1] - 0.5 * gt_loc[3]
                gt_xmax1 = gt_loc[0] + 0.5 * gt_loc[2]
                gt_ymax1 = gt_loc[1] + 0.5 * gt_loc[3]

                # gt_loc2 = grondtruth_b[j,4:8]
                gt_xmin2 = gt_loc[4] - 0.5 * gt_loc[6]
                gt_ymin2 = gt_loc[5] - 0.5 * gt_loc[7]
                gt_xmax2 = gt_loc[4] + 0.5 * gt_loc[6]
                gt_ymax2 = gt_loc[5] + 0.5 * gt_loc[7]

                gt_xmin = tf.minimum(gt_xmin1, gt_xmin2)
                gt_ymin = tf.minimum(gt_ymin1, gt_ymin2)
                gt_xmax = tf.maximum(gt_xmax1, gt_xmax2)
                gt_ymax = tf.maximum(gt_ymax1, gt_ymax2)

                iou1 = calculate_iou(anchor_xmin,anchor_ymin,anchor_xmax,anchor_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax)
                # iou2 = calculate_iou(anchor_xmin,anchor_ymin,anchor_xmax,anchor_ymax, gt_xmin2, gt_ymin2, gt_xmax2, gt_ymax2)



                loc_l = smooth_l1_loss(pred_loc,anchor_cx,anchor_cy,anchor_w,anchor_h, gt_loc)     # loc loss(positive)
                conf_l = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls,logits=pred_conf)   # conf loss(positive)
                mask1 = tf.cast(tf.greater(iou1,para.MATCH_IOU),dtype=tf.float32)
                # mask2 = mask1*tf.cast(tf.greater(iou2,para.MATCH_IOU),dtype=tf.float32)
                pos_l = pos_l + mask1*(conf_l + para.ALPHA*loc_l)
                
            flag = tf.cast(tf.equal(pos_l,0.0),dtype=tf.float32)
            loss = flag*tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls_bg,logits=pred_conf)   # conf loss(neg)
            conf_loss1 = conf_loss1.write(i,loss)
            pos_loss1 =  pos_loss1 + pos_l
            return [i+1,pos_loss1,conf_loss1]
        
        conf_loss1 = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        [i,pos_loss1,conf_loss1] = tf.while_loop(cond1,body1,loop_vars=[i,pos_loss1,conf_loss1], parallel_iterations=10,back_prop=True,swap_memory=True)
        conf_loss1 = conf_loss1.stack()


        num_pos1 = tf.cast((num_anchors-tf.count_nonzero(conf_loss1, dtype=tf.int32)),tf.float32)

        noumber_of_positives.append(num_pos1)
        positive_loss.append(pos_loss1)
        loss1 = tf.cond(tf.equal(num_pos1,0),lambda:0.0,lambda:merge_loss(pos_loss1,conf_loss1,num_pos1))

        #
        # def cond2(m, pos_loss2, conf_loss2):
        #     boolean = tf.less(m, num_anchors)
        #     return boolean
        #
        # def body2(m, pos_loss2, conf_loss2):
        #     pos_l = 0.0
        #     pred_loc = loc_b[m, :]
        #     pred_conf = cls_b[m, :]
        #     anchor = anchors_tensor[m, :]
        #
        #     anchor_cx, anchor_cy, anchor_w, anchor_h = anchor[0], anchor[1], anchor[2], anchor[3]
        #     anchor_xmin = anchor_cx - anchor_w * 0.5
        #     anchor_ymin = anchor_cy - anchor_h * 0.5
        #     anchor_xmax = anchor_cx + anchor_w * 0.5
        #     anchor_ymax = anchor_cy + anchor_h * 0.5
        #
        #     for j in range(para.MAX_NUM_GT):
        #         gt_loc = grondtruth_b[j, 4:8]
        #         gt_cls = grondtruth_b[j, 8::]
        #
        #         gt_xmin = gt_loc[0] - 0.5 * gt_loc[2]
        #         gt_ymin = gt_loc[1] - 0.5 * gt_loc[3]
        #         gt_xmax = gt_loc[0] + 0.5 * gt_loc[2]
        #         gt_ymax = gt_loc[1] + 0.5 * gt_loc[3]
        #
        #         iou = calculate_iou(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax, gt_xmin, gt_ymin, gt_xmax,
        #                             gt_ymax)
        #         pred = pred_loc[4:8]
        #         loc_l = smooth_l1_loss(pred, anchor_cx, anchor_cy, anchor_w, anchor_h, gt_loc)  # loc loss(positive)
        #         conf_l = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls, logits=pred_conf)  # conf loss(positive)
        #         mask = tf.cast(tf.greater(iou, para.MATCH_IOU), dtype=tf.float32)
        #         pos_l = pos_l + mask * (conf_l + para.ALPHA * loc_l)
        #
        #     flag = tf.cast(tf.equal(pos_l, 0.0), dtype=tf.float32)
        #     loss = flag * tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls_bg, logits=pred_conf)  # conf loss(neg)
        #     conf_loss2 = conf_loss2.write(m, loss)
        #     pos_loss2 = pos_loss2 + pos_l
        #     return [m + 1, pos_loss2, conf_loss2]
        #
        # conf_loss2 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # [m, pos_loss2, conf_loss2] = tf.while_loop(cond2, body2, loop_vars=[m, pos_loss2, conf_loss2], parallel_iterations=10,
        #                                          back_prop=True, swap_memory=True)
        # conf_loss2 = conf_loss2.stack()
        #
        # num_pos2 = tf.cast((num_anchors - tf.count_nonzero(conf_loss2, dtype=tf.int32)), tf.float32)
        #
        # noumber_of_positives.append(num_pos2)
        # positive_loss.append(pos_loss2)
        # loss2 = tf.cond(tf.equal(num_pos2, 0), lambda: 0.0, lambda: merge_loss(pos_loss2, conf_loss2, num_pos2))



        # loss = (loss1 + loss2) / 2
        losses.append(loss1)
    return tf.reduce_mean(losses), noumber_of_positives, positive_loss