# -*- coding: utf-8 -*-

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
import parameters as para
# from groundtruth import get_groundtruth
from coordinates_extractor import ground_truth_four

train_image_name = pd.read_csv(os.path.join(para.PATH,'data','train','train.txt'),header=None,names=['Name'])
val_image_name = pd.read_csv(os.path.join(para.PATH,'data','val','val.txt'),header=None,names=['Name'])

def mini_batch(i,batch_size,flag):
    if flag == 'train':
        data_name = train_image_name
        
    elif flag == 'val':
        data_name = val_image_name
        
    else:
        print('The argument "%s"  does not exist!' % (flag))
        sys.exit(0)
        
    start = (i*batch_size) % len(data_name['Name'])
    end = min(start+batch_size,len(data_name['Name']))
    
    if (end - start) < batch_size:
        start = len(data_name['Name']) - batch_size
        end = len(data_name['Name'])
    
    image = np.zeros([batch_size,para.INPUT_SIZE[0],para.INPUT_SIZE[1],para.CHANNEL],dtype=np.float32)
    groundtruth = np.zeros([batch_size,para.MAX_NUM_GT,8+para.NUM_CLASSESS],dtype=np.float32)
       
    batch_name = np.array(data_name['Name'][start:end])
    for j in range(len(batch_name)):
        image_name = os.path.join(para.PATH,'data','annotation','images',batch_name[j]+'.'+para.PIC_TYPE) #  persian
        # image_name = os.path.join(para.PATH,'data','ccpd_base',batch_name[j]+'.'+para.PIC_TYPE)
        # print(image_name)
        im = cv.imread(image_name).astype(np.float32)
        image[j,:,:,:] = cv.resize(im,(para.INPUT_SIZE[0],para.INPUT_SIZE[1])).astype(np.float32)
        
        xml_name = os.path.join(para.PATH,'data','annotation','xml',batch_name[j]+'.xml') # persian
        # xml_name = image_name
        # gt = get_groundtruth(xml_name) # persian
        gt = ground_truth_four(xml_name)
        groundtruth[j,:,:] = gt['groundtruth']
    return {'image':image,'groundtruth':groundtruth,'image_name':batch_name,'image_num':data_name.shape[0]}


# print(mini_batch(0, 4, 'train'))