from __future__ import print_function, division
import numpy as np
import datetime
import os, sys
from scipy.misc import imread
import cv, cv2


def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    train_edge_names=[]
    val_input_names=[]
    val_output_names=[]
    val_edge_names=[]
    test_input_names=[]
    test_output_names=[]
    test_edge_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/train_edge_labels"):
        cwd = os.getcwd()
        train_edge_names.append(cwd + "/" + dataset_dir + "/train_edge_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/val_edge_labels"):
        cwd = os.getcwd()
        val_edge_names.append(cwd + "/" + dataset_dir + "/val_edge_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    for file in os.listdir(dataset_dir + "/test_edge_labels"):
        cwd = os.getcwd()
        test_edge_names.append(cwd + "/" + dataset_dir + "/test_edge_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort(), train_edge_names.sort(), val_edge_names.sort(), test_edge_names.sort(),
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names, train_edge_names, val_edge_names, test_edge_names

# Input images & GT boundaries -- gray-scale images
def load_image1(path):
    image = cv2.cvtColor(cv2.imread(path,0), cv.CV_GRAY2RGB)
    return image

# GT -- color images
def load_image2(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

# Resize the image
def resize_data(image, label, resize_height, resize_width):
    image = cv2.resize(image, (resize_height, resize_width))
    label = cv2.resize(label, (resize_height, resize_width))
    return image, label

def compute_cpa(pred, label, num_classes, resize_height, resize_width):
    pred = np.reshape(pred,(resize_height, resize_width))
    label = np.reshape(label, (resize_height, resize_width))
    total = [0.0] * num_classes
    for val in range(num_classes):
        total[val] = (label==val).sum()

    count = [0.0] * num_classes
    for i in range(num_classes):
        idx,idy = np.where(pred==i)
        for j in range(len(idx)):
            xx = idx[j]
            yy = idy[j]
            if pred[xx,yy] == label[xx,yy]:
                count[i] = count[i] + 1.0

    accuracies = []
    for i in range(len(total)):
        # Remove noises in GT
        if total[i] <= 20.1:
            accuracies.append(8.8)
        else:
            accuracies.append(count[i] / total[i])
    return accuracies


def compute_iou(pred, label, num_classes, resize_height, resize_width):
    pred = np.reshape(pred,(resize_height, resize_width))
    label = np.reshape(label, (resize_height, resize_width))
    total1 = [0.0] * num_classes
    total2 = [0.0] * num_classes
    for val in range(num_classes):
        total1[val] = (label==val).sum()
        total2[val] = (pred==val).sum()
    count = [0.0] * num_classes
    for i in range(num_classes):
        idx,idy = np.where(pred==i)
        for j in range(len(idx)):
            xx= idx[j]
            yy = idy[j]
            if pred[xx,yy] == label[xx,yy]:
                count[i] = count[i] + 1.0

    IoU = []
    total = np.subtract(np.add(total1, total2), count)
    for i in range(len(total)):
        if total1[i] <= 20.1:
            IoU.append(8.8)
        else:
            IoU.append(count[i] / total[i])
    return IoU


def evaluate_segmentation(pred, label, num_classes, resize_height, resize_width):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    cpa = compute_cpa(flat_pred, flat_label, num_classes, resize_height, resize_width)
    iou = compute_iou(flat_pred, flat_label, num_classes, resize_height, resize_width)

    return cpa, iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

