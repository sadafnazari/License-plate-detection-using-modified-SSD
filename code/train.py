# -*- coding: utf-8 -*-

from __future__ import division

import os
import sys
import shutil
import tensorflow as tf
import parameters as para
from readbatch import mini_batch
from SSDLoss import ssd_loss
from SSDNet import SSDmobilenetv1
from postprocessing import nms,box_decode,save_instance,output_generator
from SSDLossTest import validation_loss
import matplotlib.pyplot as plt


def net_placeholder(batch_size=None):
    Input = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size, para.INPUT_SIZE[0], para.INPUT_SIZE[1], para.CHANNEL], name='Input')

    groundtruth = tf.placeholder(dtype=tf.float32,
                            shape=[batch_size, para.MAX_NUM_GT, 8+para.NUM_CLASSESS], name='Label')
    
    isTraining = tf.placeholder(tf.bool, name='Batch_norm')
    return Input, groundtruth, isTraining


def training_net():
    image, groundtruth, isTraining = net_placeholder(batch_size=None)
    loc, cls = SSDmobilenetv1(image, isTraining)
    # loss = ssd_loss(loc, cls, groundtruth)
    loss, num, p_loss = ssd_loss(loc, cls, groundtruth)
    # loss = ssd_loss(loc, cls, groundtruth)
    train_losses = []
    valid_losses = []
    test_losses = []
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(para.LEARNING_RATE1).minimize(loss)
     
    Saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=5)
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter(os.path.join(para.PATH,'model'), sess.graph)
        init_var_op = tf.global_variables_initializer()
        sess.run(init_var_op)
        
        # restore model 
        if para.RESTORE_MODEL:
            if not os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH):
                print('Model does not exist！')
                sys.exit()
            ckpt = tf.train.get_checkpoint_state(para.CHECKPOINT_MODEL_SAVE_PATH)
            model = ckpt.model_checkpoint_path.split('/')[-1]
            Saver.restore(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH, model))
            print('Successfully restore model:',model)
        
        for i in range(para.TRAIN_STEPS):
            batch = mini_batch(i,para.BATCH_SIZE,'train')
            # print(batch['groundtruth'])
            feed_dict = {image:batch['image'], groundtruth:batch['groundtruth'], isTraining:True}
            # _, loss_= sess.run([train_step,loss],feed_dict=feed_dict)
            _, loss_, n_, p_= sess.run([train_step,loss, num, p_loss],feed_dict=feed_dict)
            print('===>Step %d: training loss = %g' % (i,loss_))
            print('number of matched: ', n_, ' positive los: ', p_)
            train_losses.append(loss_)

            # evaluate the training samples
            if i%250 == 0:
                valid_loss = 0
                feed_dict_show = {image: batch['image'], groundtruth:batch['groundtruth'], isTraining: False}
                location_, confidence_ = sess.run([loc, cls], feed_dict=feed_dict_show)
                # print(confidence_)
                for j in range(para.BATCH_SIZE):
                    predictions = {'location': location_[j], 'confidence': confidence_[j]}
                    # print('image name: ', len(batch['groundtruth'][j]))
                    # pred_output = box_decode(predictions, batch['image_name'][j])
                    pred_output = output_generator(predictions, batch['image_name'][j])
                    pred_output = nms(pred_output, para.NMS_THRESHOLD)
                    # print(batch['groundtruth'])
                    save_instance(pred_output, batch['groundtruth'][j][0], False)
                    # print(batch['groundtruth'])
                    # valid_loss += validation_loss(batch['groundtruth'][j][0], location_[j], confidence_[j])
                # valid_loss /= para.BATCH_SIZE
                # print('validation loss: ', valid_loss)
                # valid_losses.append(valid_loss)

            # evaluate and save checkpoint
            if i % 250 == 0:
                write_instance_dir = os.path.join(para.PATH,'pic')
                if not os.path.exists(write_instance_dir):os.mkdir(write_instance_dir)
                j = 0
                test_loss = 0.0

                while True:
                    batch = mini_batch(j,1,'val')
                    feed_dict = {image:batch['image'], groundtruth:batch['groundtruth'],isTraining:False}
                    location,confidence = sess.run([loc,cls],feed_dict=feed_dict)
                    predictions = {'location':location,'confidence':confidence}
                    pred_output = output_generator(predictions,batch['image_name'])
                    pred_output = nms(pred_output,para.NMS_THRESHOLD)
                    # print(len(location))
                    # test_loss += validation_loss(batch['groundtruth'][0][0], location[0], confidence[0])
                    if j < min(5,batch['image_num']):save_instance(pred_output, batch['groundtruth'][0][0], True)
                    if j == batch['image_num']-1:
                        # test_loss /= 10
                        # test_losses.append(test_loss)
                        break
                    j += 1
                if os.path.exists(para.CHECKPOINT_MODEL_SAVE_PATH) and (i==0):
                    shutil.rmtree(para.CHECKPOINT_MODEL_SAVE_PATH)
                Saver.save(sess,os.path.join(para.CHECKPOINT_MODEL_SAVE_PATH,para.MODEL_NAME))

    plt.plot(range(len(train_losses)), train_losses, 'r', label='Training loss')
    # plt.plot(range(len(valid_losses)), valid_losses, 'g', label='Validation loss')
    # plt.plot(range(len(test_losses)), test_losses, 'b', label='Test loss')
    plt.title('Training loss')
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    # plt.legend()
    plt.figure()
    plt.show()
            

def main():
    training_net()
     
if __name__ == '__main__':
    main()
