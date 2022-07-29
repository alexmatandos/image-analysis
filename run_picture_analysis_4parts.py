import pandas
import os
import glob

import imageio

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import math
import numpy

import kfold_template

category_dataset = pandas.read_csv("pictures_category.csv")
# print(category_dataset)



def getrgb(file_path):
	imimage = imageio.imread(file_path, pilmode="RGB")
	imimage = imimage/255

	imimage_top_left = imimage[0:math.ceil(imimage.shape[0]/2),0:math.ceil(imimage.shape[1]/2)]
	imimage_top_right = imimage[0:math.ceil(imimage.shape[0]/2),math.ceil(imimage.shape[1]/2):imimage.shape[1]]
	imimage_bottom_left = imimage[math.ceil(imimage.shape[0]/2):imimage.shape[0],0:math.ceil(imimage.shape[1]/2)]
	imimage_bottom_right = imimage[math.ceil(imimage.shape[0]/2):imimage.shape[0],math.ceil(imimage.shape[1]/2):imimage.shape[1]]

	imimage_top_left = imimage_top_left.sum(axis = 0).sum(axis = 0)/(imimage_top_left.shape[0]*imimage_top_left.shape[1])
	imimage_bottom_left = imimage_bottom_left.sum(axis = 0).sum(axis = 0)/(imimage_bottom_left.shape[0]*imimage_bottom_left.shape[1])
	imimage_top_right = imimage_top_right.sum(axis = 0).sum(axis = 0)/(imimage_top_right.shape[0]*imimage_top_right.shape[1])
	imimage_bottom_right = imimage_bottom_right.sum(axis = 0).sum(axis = 0)/(imimage_bottom_right.shape[0]*imimage_bottom_right.shape[1])


	imimage = numpy.concatenate((imimage_top_left, imimage_top_right, imimage_bottom_left, imimage_bottom_right))
	return imimage

# print(getrgb("pictures/pic01.jpeg"))


def read_picture_folder(folder_name):
	result = pandas.DataFrame()
	for file_path in glob.glob(folder_name + "/*"):
		image_features = pandas.DataFrame(getrgb(file_path))
		image_features = pandas.DataFrame.transpose(image_features)
		image_features["filename"] = file_path.replace(folder_name + "/","")
		result = pandas.concat([result, image_features])
	result = result.rename(columns={0: "red", 1:"green", 2:"blue"})
	return result	

image_dataset = read_picture_folder("pictures/pictures")

# print(image_dataset)

dataset = pandas.merge(image_dataset, category_dataset, on="filename")

dataset.to_csv("4parts.csv")

