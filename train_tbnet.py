from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import argparse
import random
import os
import cv2

from utils import utils, helpers
from builders import model_builder


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--dataset', type=str, default="AirportPavementDisease", help='The dataset to use.')
parser.add_argument('--resize_height', type=int, default=512, help='Height of resized input image to network')
parser.add_argument('--resize_width', type=int, default=512, help='Width of resized input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to use for validations')
args = parser.parse_args()


# Get the names of the classes from the class dictionary file
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
net_edge = tf.placeholder(tf.float32,shape=[None,None])

network, init_fn, output_edge = model_builder.build_model(model_name="tbnet", frontend="ResNet101", net_input=net_input, num_classes=num_classes, is_training=True)

# The softmax cross entropy loss with the weighting mechanism
print("Computing class weights for", args.dataset, "...")
class_weights = utils.compute_class_weights(labels_dir=args.dataset + "/train_labels", label_values=label_values)
weights = tf.reduce_sum(class_weights * net_output, axis=-1)
unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
weighted_loss = unweighted_loss * weights
loss_1 = tf.reduce_sum(weighted_loss)

# The binary cross entropy loss
loss_2 = tf.reduce_sum(tf.keras.losses.binary_crossentropy(net_edge, output_edge))

loss = loss_1 + loss_2

opt = tf.train.RMSPropOptimizer(learning_rate=0.00001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=300)
sess.run(tf.global_variables_initializer())


if init_fn is not None:
    init_fn(sess)

model_checkpoint_name = "checkpoints/latest_model_" + args.dataset + ".ckpt"

# Load the data
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names, train_edge_names, val_edge_names, test_edge_names = utils.prepare_data(dataset_dir=args.dataset)


avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

# Do the training
for epoch in range(0, args.num_epochs):

    current_losses = []

    cnt=0

    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):

        input_image_batch = []
        output_image_batch = []

        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            
            input_image = utils.load_image1(train_input_names[id])
            output_image = utils.load_image2(train_output_names[id])
            edge_image = utils.load_image1(train_edge_names[id])
            edge_image = cv2.resize(edge_image,(args.resize_height, args.resize_width))
            edge_image = np.float32(edge_image) / 255.0

            input_image, output_image = utils.resize_data(input_image, output_image, args.resize_height, args.resize_width)

            # Prepare the data
            input_image = np.float32(input_image) / 255.0
            output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

            input_image_batch.append(np.expand_dims(input_image, axis=0))
            output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))


        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch,net_edge:edge_image[:,:,0]})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    print("Average loss per epoch # %04d:" % (mean_loss))
    avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
        os.makedirs("%s/%04d"%("checkpoints",epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if epoch % 1 == 0:
         print("Saving checkpoint for this epoch")
         saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))

         class_scores_list = []
         iou_list = []

         for ind in val_indices:
             input_image = utils.load_image1(val_input_names[ind])
             input_image,_ = utils.resize_data(input_image, input_image, args.resize_height, args.resize_width)
             input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0
             gt = utils.load_image2(val_output_names[ind])
             gt,_ = utils.resize_data(gt, gt, args.resize_height, args.resize_width)
             gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
        
             output_image = sess.run(network,feed_dict={net_input:input_image})
        
             output_image = np.array(output_image[0,:,:,:])
             output_image = helpers.reverse_one_hot(output_image)
             out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        
             class_accuracies, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes, resize_height=args.resize_height, resize_width=args.resize_width)
             class_scores_list.append(class_accuracies)
             gt = helpers.colour_code_segmentation(gt, label_values)
             iou_list.append(iou)

         class_scores_list_new=map(list,zip(*class_scores_list))
         avg_pre=[]
         new=class_scores_list_new
         for i in range(len(class_scores_list_new)):
             # Remove noises in GT
             new[i]=np.delete(new[i],np.where(new[i]==8.8))
             new[i]=np.delete(new[i],np.where(new[i]==8.8))
             if class_scores_list_new[i].size==0:
                 continue
             else:
                 avg_pre.append(np.mean(new[i]))
         print("Average accuracy for epoch # %04d = %f" %(epoch, np.mean(avg_pre[1:])))
         avg_scores_per_epoch.append(np.mean(avg_pre[1:]))

         class_iou_list=map(list,zip(*iou_list))
         avg_iou=[]
         new_=class_iou_list
         for i in range(len(class_iou_list)):
             new_[i]=np.delete(new_[i],np.where(new_[i]==8.8))
             new_[i]=np.delete(new_[i],np.where(new_[i]==8.8))
             if new_[i].size!=0:
                 avg_iou.append(np.mean(new_[i]))
         print("Average IoU for epoch # %04d = %f" %(epoch, np.mean(avg_iou[1:])))
         avg_iou_per_epoch.append(np.mean(avg_iou[1:]))


    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)


