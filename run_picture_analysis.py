import pandas
import os
import glob
import imageio
from sklearn import linear_model
import kfold_template
from sklearn.ensemble import RandomForestClassifier

category_dataset = pandas.read_csv("pictures_category.csv")

#file_path = "pictures/pictures/pic01.jpeg"
#imimage = imageio.imread(file_path, pilmode = "RGB")
#imimage = imimage/255
##generating the average RGB composition for the image
#imimage = imimage.sum(axis = 0).sum(axis = 0)/(imimage.shape[0]*imimage.shape[1])
#print(imimage)

#'imread' generates matrices from each pixel in each line containing the RGB data in it
#define a function from the above

def get_rgb(file_path):
	imimage = imageio.imread(file_path, pilmode = "RGB")
	imimage = imimage/255
	imimage = imimage.sum(axis = 0).sum(axis = 0)/(imimage.shape[0]*imimage.shape[1])
	return imimage

#results = pandas.DataFrame()

#for file_path in glob.glob("pictures/pictures/*"):
#	#print(file_path)
#	image_features = pandas.DataFrame(get_rgb(file_path))
#	image_features = pandas.DataFrame.transpose(image_features)
#	results = pandas.concat([results, image_features])

def read_picture_folder(folder_name):
	results = pandas.DataFrame()
	for file_path in glob.glob(folder_name + "/*"):
		print(file_path)
		image_features = pandas.DataFrame(get_rgb(file_path))
		image_features = pandas.DataFrame.transpose(image_features)
		image_features['filename'] = file_path.replace(folder_name + "\\", "")
		results = pandas.concat([results, image_features])
	results = results.rename(columns = {0: "Red", 1: "Green", 2: "Blue"})
	return results

image_dataset = read_picture_folder("pictures/pictures")
#print(image_dataset)

dataset = pandas.merge(image_dataset, category_dataset, on = 'filename')
#print(dataset)

data = dataset.iloc[:, 0:3].values
target = dataset.iloc[:, 4].factorize()
target_index = target[1]
target = target[0]


machine = linear_model.LogisticRegression()
result = kfold_template.run_kfold(data, target, 4, machine, 1, 1 ,1)

#printing accuracy scores and the confusion matrices
print(result[1])
for i in result[2]:
	print(i)

machine = linear_model.LogisticRegression()
machine.fit(data, target)

new_image_dataset = read_picture_folder("new_pictures/new_pictures")
prediction = machine.predict(new_image_dataset.iloc[:, 0:3])
prediction = list(target_index[prediction])
new_image_dataset['prediction_logistic_regression'] = prediction
#print(new_image_dataset)
#logistic regression will predict very badly!!

machine2 = RandomForestClassifier(criterion = "gini", max_depth = 10, n_estimators = 300 ,bootstrap = True, max_features = "auto")
result2 = kfold_template.run_kfold(data, target, 4, machine2, 1, 1, 1)

print(result2[1])
for x in result2[2]:
	print(x)

prediction2 = machine2.predict(new_image_dataset.iloc[:, 0:3])
prediction2 = list(target_index[prediction2])
new_image_dataset['prediction_random_forest'] = prediction2
print(new_image_dataset)

#adding a picture with some "disturbance" (such as featured in new_pic06.jpeg), the prediction might be messed up, even though the machine is pretty good!