import sys, os
import subprocess

sys.path.append("models")

from models.tbnet import build_tbnet


def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])


def build_model(model_name, net_input, num_classes, frontend="ResNet101", is_training=True):
	print("Preparing the model ...")

	if not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("ResNet101")

        network, init_fn, output_edge = build_tbnet(net_input, frontend=frontend, num_classes=num_classes, is_training=is_training)

	return network, init_fn, output_edge
