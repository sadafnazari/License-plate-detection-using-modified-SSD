# -*- coding: utf-8 -*-

from __future__ import division

import os
import math
import datetime
import cv2 as cv
import numpy as np
import parameters as para
from onehotcode import onehotdecode


def softmax(logits):
    return np.exp(logits)/np.sum(np.exp(logits))


def correct_location(x):
    x = min(max(x,0),1)
    return x


def output_generator(predictions, imgname):
    location = np.squeeze(predictions['location'])
    confidence = np.squeeze(predictions['confidence'])
    boxes = []
    for i in range(len(confidence)):
        pred_box = location[i,:]
        pred_conf = confidence[i,:]
        anchor = para.ANCHORS.loc[i]
        conf_prob = softmax(pred_conf)
        max_prob = max(conf_prob)
        pred_class = onehotdecode(conf_prob)

        if pred_class == para.LABELS.Class_name[para.NUM_CLASSESS - 1]:  # ignored background category
            continue

        if max_prob > 0.6:
            pred_box_cx1 = pred_box[0]
            pred_box_cy1 = pred_box[1]
            pred_box_w1 = pred_box[2]
            pred_box_h1 = pred_box[3]


            pred_box_cx2 = pred_box[4]
            pred_box_cy2 = pred_box[5]
            pred_box_w2 = pred_box[6]
            pred_box_h2 = pred_box[7]

            cx1 = (pred_box_cx1 / para.X_SCALE) * anchor.w + anchor.cx
            cy1 = (pred_box_cy1 / para.Y_SCALE) * anchor.h + anchor.cy
            w1 = math.exp(pred_box_w1 / para.W_SCALE) * anchor.w
            h1 = math.exp(pred_box_h1 / para.H_SCALE) * anchor.h

            x1 = correct_location(cx1 - 0.5 * w1)
            y1 = correct_location(cy1 - 0.5 * h1)

            x3 = correct_location(cx1 + 0.5 * w1)
            y3 = correct_location(cy1 + 0.5 * h1)

            cx2 = (pred_box_cx2 / para.X_SCALE) * anchor.w + anchor.cx
            cy2 = (pred_box_cy2 / para.Y_SCALE) * anchor.h + anchor.cy
            w2 = math.exp(pred_box_w2 / para.W_SCALE) * anchor.w
            h2 = math.exp(pred_box_h2 / para.H_SCALE) * anchor.h

            x2 = correct_location(cx2 + 0.5 * w2)
            y2 = correct_location(cy2 - 0.5 * h2)

            x4 = correct_location(cx2 - 0.5 * w2)
            y4 = correct_location(cy2 + 0.5 * h2)

            box = {'box': [x1, y1, x2, y2, x3, y3, x4, y4, max_prob], 'className': pred_class}
            # box = {'box':[xmin, ymin, xmax, ymax,max_prob],'className':pred_class}
            boxes.append(box)

    result = {'imageName': imgname, 'boxes': boxes}
    return result


def box_decode(predictions,imgname):
    location = np.squeeze(predictions['location'])
    confidence = np.squeeze(predictions['confidence'])
    boxes = []
    for i in range(len(para.ANCHORS)):
        pred_box = location[i,:]
        pred_conf = confidence[i,:]
        anchor = para.ANCHORS.loc[i]

        pred_box_cx1 = pred_box[0]
        pred_box_cy1 = pred_box[1]
        pred_box_w1 = pred_box[2]
        pred_box_h1 = pred_box[3]

        cx1 = (pred_box_cx1/para.X_SCALE)*anchor.w + anchor.cx
        cy1 = (pred_box_cy1 / para.Y_SCALE) * anchor.h + anchor.cy
        w1 = math.exp(pred_box_w1 / para.W_SCALE) * anchor.w
        h1 = math.exp(pred_box_h1/para.H_SCALE)*anchor.h

        x1 = correct_location(cx1 - 0.5 * w1)
        y1 = correct_location(cy1 - 0.5 * h1)

        x3 = correct_location(cx1 + 0.5 * w1)
        y3 = correct_location(cy1 + 0.5 * h1)

        conf_prob = softmax(pred_conf)
        max_prob = max(conf_prob)
        pred_class = onehotdecode(conf_prob)

        if pred_class == para.LABELS.Class_name[para.NUM_CLASSESS - 1]:  # ignored background category
            continue

        for j in range(len(para.ANCHORS)):

            pred_box_cx2 = pred_box[4]
            pred_box_cy2 = pred_box[5]
            pred_box_w2 = pred_box[6]
            pred_box_h2 = pred_box[7]



            cx2 = (pred_box_cx2/para.X_SCALE)*anchor.w + anchor.cx
            cy2 = (pred_box_cy2/para.Y_SCALE)*anchor.h + anchor.cy
            w2 = math.exp(pred_box_w2/para.W_SCALE)*anchor.w
            h2 = math.exp(pred_box_h2/para.H_SCALE)*anchor.h


            x2 = correct_location(cx2 + 0.5 * w2)
            y2 = correct_location(cy2 - 0.5 * h2)

            x4 = correct_location(cx2 - 0.5 * w2)
            y4 = correct_location(cy2 + 0.5 * h2)

            conf_prob2 = softmax(pred_conf)
            max_prob2 = max(conf_prob2)
            pred_class2 = onehotdecode(conf_prob2)

            if pred_class2 == para.LABELS.Class_name[para.NUM_CLASSESS - 1]:  # ignored background category
                continue

            max_prob3 = (max_prob2 + max_prob) /2

            box = {'box':[x1,y1,x2,y2,x3,y3,x4,y4,max_prob3],'className':pred_class}
        # box = {'box':[xmin, ymin, xmax, ymax,max_prob],'className':pred_class}
        boxes.append(box)
                    
    result = {'imageName':imgname,'boxes':boxes}
    return result


def calculateIoU(xmin0,ymin0,xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    intersection = w*h
    union = (xmax0-xmin0)*(ymax0-ymin0)+(xmax1-xmin1)*(ymax1-ymin1)-intersection
    
    if union<=0.0:
        iou = 0.0
    else:
        iou = 1.0*intersection / union
    return iou


def nms(result,threshold):
    class_list =[]
    final_pred_boxes = []
    boxes = result['boxes']
    
    for b in range(len(boxes)):
        class_list.append(boxes[b]['className'])
    class_list = np.unique(class_list)
    
    for name in class_list:
        box_coord = []
        for b in range(len(boxes)):
            if name == boxes[b]['className']:
                box_coord.append(boxes[b]['box'])       
        box_coord = np.array(box_coord)
        
        while box_coord.shape[0] > 0:
            idx = np.argmax(box_coord[:,-1])
            keep_box = box_coord[idx,:]
            pred_box = {'box':keep_box,'className':name}
            final_pred_boxes.append(pred_box)

            box_coord = np.delete(box_coord,[idx],axis=0)
            if box_coord.shape[0] == 0:break

            suppre = []
            xmin0 = min(keep_box[0], keep_box[6])
            ymin0 = min(keep_box[1], keep_box[3])
            xmax0 = min(keep_box[2], keep_box[4])
            ymax0 = min(keep_box[5], keep_box[7])

            for b in range(box_coord.shape[0]):
                xmin1 = min(box_coord[b,:][0], box_coord[b,:][6])
                ymin1 = min(box_coord[b,:][1], box_coord[b,:][3])
                xmax1 = min(box_coord[b,:][2], box_coord[b,:][4])
                ymax1 = min(box_coord[b,:][5], box_coord[b,:][7])

                iou = calculateIoU(xmin0,ymin0,xmax0,ymax0,
                                   xmin1,ymin1,xmax1,ymax1)
                if iou > threshold:
                    suppre.append(b)
            box_coord = np.delete(box_coord,suppre,axis=0)
    detections = {'imageName':result['imageName'],'boxes':final_pred_boxes}
    return detections


def save_instance(detections, batch, val):
    if val:
        image_name = detections['imageName'][0]+'.'+para.PIC_TYPE
    else:
        image_name = detections['imageName']+'.'+para.PIC_TYPE
    read_dir = os.path.join(para.PATH,'data','annotation','images', image_name)
    # print(read_dir)
    if val:
        write_dir = os.path.join(para.PATH,'pic', 'val')
    else:
        write_dir = os.path.join(para.PATH, 'pic', 'train')
    
    im = cv.imread(read_dir).astype(np.float32)
    im_h = im.shape[0]
    im_w = im.shape[1]
    
    im = cv.resize(im,(para.INPUT_SIZE[0],para.INPUT_SIZE[1])).astype(np.float32)
    for b in range(len(detections['boxes'])):
        box = detections['boxes'][b]['box']
        name = detections['boxes'][b]['className']
        
        x1 = int(box[0]*para.INPUT_SIZE[1])
        y1 = int(box[1]*para.INPUT_SIZE[0])
        x2 = int(box[2]*para.INPUT_SIZE[1])
        y2 = int(box[3]*para.INPUT_SIZE[0])
        x3 = int(box[4]*para.INPUT_SIZE[1])
        y3 = int(box[5]*para.INPUT_SIZE[0])
        x4 = int(box[6]*para.INPUT_SIZE[1])
        y4 = int(box[7]*para.INPUT_SIZE[0])
        prob = min(round(box[8]*100),100.0)
        txt = name +':'+ str(prob) + '%'


        
        font = cv.FONT_HERSHEY_PLAIN
        # im = cv.rectangle(im,(xmin,ymin),(xmax,ymax),(255, 0, 0),1)
        im = cv.line(im, (x1,y1), (x2,y2), (255, 0, 0), 1)
        im = cv.line(im, (x2,y2), (x3,y3), (255, 0, 0), 1)
        im = cv.line(im, (x3,y3), (x4,y4), (255, 0, 0), 1)
        im = cv.line(im, (x4,y4), (x1,y1), (255, 0, 0), 1)
        im = cv.putText(im,txt,(x1,y1),font,0.5,(255,0,0),1)

    # print(batch)
    # if val:
    #     print(batch)
    # for i in range(len(batch)):
    #     if i%2 == 0:
    #         batch2[i] = batch[i]*300
    #     else:
    #         batch2[i] = batch[i]*300


    g_x1 = int((batch[0] - 0.5 * batch[2]) * 300)
    g_y1 = int((batch[1] - 0.5 * batch[3]) * 300)

    g_x2 = int((batch[4] + 0.5 * batch[6]) * 300)
    g_y2 = int((batch[5] - 0.5 * batch[7]) * 300)

    g_x3 = int((batch[0] + 0.5 * batch[2]) * 300)
    g_y3 = int((batch[1] + 0.5 * batch[3]) * 300)

    g_x4 = int((batch[4] - 0.5 * batch[6]) * 300)
    g_y4 = int((batch[5] + 0.5 * batch[7]) * 300)


    # im = cv.line(im, (g_x1, g_y1), (g_x2, g_y2), (0, 255, 0), 1)
    # im = cv.line(im, (g_x2, g_y2), (g_x3, g_y3), (0, 255, 0), 1)
    # im = cv.line(im, (g_x3, g_y3), (g_x4, g_y4), (0, 255, 0), 1)
    # im = cv.line(im, (g_x4, g_y4), (g_x1, g_y1), (0, 255, 0), 1)
    
    im = cv.resize(im,(im_w,im_h)).astype(np.float32)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-')
    dst = os.path.join(write_dir,current_time+image_name)
    cv.imwrite(dst,im)
