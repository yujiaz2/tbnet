import os,time,cv2, sys
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the checkpoint weights.')
parser.add_argument('--resize_height', type=int, default=512, help='Height of resized input image to network')
parser.add_argument('--resize_width', type=int, default=512, help='Width of resized input image to network')
parser.add_argument('--dataset', type=str, default="AirportPavementDisease", required=False, help='The dataset to use')
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

# Initializing the network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 
net_edge = tf.placeholder(tf.float32,shape=[None,None]) 

network, _, output_edge = model_builder.build_model("tbnet", net_input=net_input, num_classes=num_classes, is_training=False)

sess.run(tf.global_variables_initializer())

# Load the weights from the checkpoint
saver=tf.train.Saver(max_to_keep=300)
saver.restore(sess, args.checkpoint_path)

# Load the data
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names, train_edge_names, val_edge_names, test_edge_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s"%("Results")):
        os.makedirs("%s"%("Results"))

class_scores_list = []
iou_list = []
run_times_list = []

# Run testing on the test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    # Prepare the data
    input_image = utils.load_image1(test_input_names[ind])
    input_image,_ = utils.resize_data(input_image,input_image,args.resize_height,args.resize_width)
    input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0
    gt = utils.load_image2(test_output_names[ind])
    gt,_ = utils.resize_data(gt,gt,args.resize_height,args.resize_width)
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
    edge_image = utils.load_image1(train_edge_names[ind])
    edge_image = cv2.resize(edge_image,(args.resize_height, args.resize_width))
    edge_image = np.float32(edge_image) / 255.0

    st = time.time()
    output_image, edge_image = sess.run([network,output_edge],feed_dict={net_input:input_image,net_edge:edge_image[:,:,0]})

    run_times_list.append(time.time()-st)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    class_accuracies, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes, resize_height=args.resize_height, resize_width=args.resize_width)

    file_name = utils.filepath_to_name(test_input_names[ind])

    class_scores_list.append(class_accuracies)
    iou_list.append(iou)
    
    gt = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s/%s_pred.png"%("Results", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%("Results", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
    edge = np.float32(edge_image) * 255.0
    cv2.imwrite("%s/%s_edge.png"%("Results", file_name), edge)

print("Computing accuracy......")
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
        print("%d %f" % (i,np.mean(new[i])))
print("Average accuracy = ", np.mean(avg_pre[1:]))

print("Computing IoU......")
class_iou_list=map(list,zip(*iou_list))
avg_iou=[]
new_=class_iou_list
for i in range(len(class_iou_list)):
    new_[i]=np.delete(new_[i],np.where(new_[i]==8.8))
    new_[i]=np.delete(new_[i],np.where(new_[i]==8.8))
    if new_[i].size!=0:
        avg_iou.append(np.mean(new_[i]))
        print("%d %f" %(i,np.mean(new_[i])))
print("Average IoU = ", np.mean(avg_iou[1:]))
