import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="ALL", help='Which model weights to download')
args = parser.parse_args()


if args.model == "ResNet101" or args.model == "ALL":
	subprocess.check_output(['wget','http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz', "-P", "models"])
	try:
		subprocess.check_output(['tar', '-xvf', 'models/resnet_v2_101_2017_04_14.tar.gz', "-C", "models"])
		subprocess.check_output(['rm', 'models/resnet_v2_101_2017_04_14.tar.gz'])
	except Exception as e:
		print(e)
		pass

