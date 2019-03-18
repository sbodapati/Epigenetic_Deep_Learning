import numpy as np
import os


def to_np(tensor):
	return tensor.data.numpy()


def make_dir(path):
	if not os.path.isdir(path):
		os.makedirs(path)

