import pickle
import os
import shutil
import random
import matplotlib.pyplot as plt

##########################################################################################
# Create dir for the Dataset
##########################################################################################

if not os.path.exists("Data_mini"):
    os.mkdir("Data_mini")
if not os.path.exists("Data_mini/images"):
    os.mkdir("Data_mini/images")

if not os.path.exists("Data_mini/images/mycoco_train2017"):
    os.mkdir("Data_mini/images/mycoco_train2017")
if not os.path.exists("Data_mini/images/mycoco_val2017"):
    os.mkdir("Data_mini/images/mycoco_val2017")

##########################################################################################
# Create a Dataset_mini for the train
##########################################################################################

var_train = os.listdir("Data/train2017")
random.seed(1234)
random.shuffle(var_train)
for diri in var_train[:19480]:
    shutil.copy("Data/train2017/" + diri, "Data_mini/images/mycoco_train2017/" + diri)

##########################################################################################
# Create a Dataset_mini for the val
##########################################################################################
var_val = os.listdir("Data/val2017")
for diri in var_val:
    shutil.copy("Data/val2017/" + diri, "Data_mini/images/mycoco_val2017/" + diri)


dir_test = "Data_mini/images/mycoco_val2017"
var = os.listdir(dir_test)

print("total root files in mycoco_val2017: ", len(var))

for diri in var:
    img = plt.imread(os.path.join(dir_test,diri))
    if len(img.shape) == 2:
        os.remove(os.path.join(dir_test,diri))

print("total root files in mycoco_val2017 after removing: ", len(os.listdir("Data_mini/images/mycoco_val2017")))

dir_train= "Data_mini/images/mycoco_train2017"
var_1 = os.listdir(dir_train)

print("total root files in mycoco_train2017: ", len(var_1))

for diri in var_1: 
    img = plt.imread(os.path.join(dir_train,diri))
    if len(img.shape) == 2:
        os.remove(os.path.join(dir_train,diri))

print("total root files in mycoco_train2017 after removing: ", len(os.listdir("Data_mini/images/mycoco_train2017")))