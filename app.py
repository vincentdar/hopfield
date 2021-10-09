import numpy as np
import random
import matplotlib.pyplot as plt
from skimage import io, color
from PIL import Image
import os
import re

def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1

def create_W(x):
    if len(x.shape) != 1:
        print("The input is not vector")
        return
    else:
        w = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(i,len(x)):
                if i == j:
                    w[i,j] = 0
                else:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w

def readImg(file, threshold):
    img = Image.open(file).convert(mode="L")
    imgArray = np.asarray(img)
    x = np.zeros(imgArray.shape)
    x[imgArray > threshold] = 1
    x[x == 0] = -1
    
    return x

def array2img(test_data, predict_data, train_data):
    y = np.zeros(test_data.shape)
    y[test_data == 1] = 255
    y[test_data == -1] = 0

    test_img = Image.fromarray(y)

    y = np.zeros(predict_data.shape)
    y[predict_data == 1] = 255
    y[predict_data == -1] = 0

    predict_img = Image.fromarray(y)

    y = np.zeros(train_data.shape)
    y[train_data == 1] = 255
    y[train_data == -1] = 0

    train_img = Image.fromarray(y)

    figure, axis = plt.subplots(1, 3)
    axis[0].imshow(test_img)
    axis[0].set_title("Tes Image")
    axis[1].imshow(predict_img)
    axis[1].set_title("Predict Image")
    axis[2].imshow(train_img)
    axis[2].set_title("Train Image")
    plt.show()
    

    return None

def update(w, y_vec, theta=0.5, time=100):
    for i in range(time):
        length = len(y_vec)
        column = random.randint(0, length-1)
        print(w[column][:].shape)
        print(y_vec.shape)
        dot_product = np.dot(w[column][:], y_vec) - theta
        
        print(dot_product.shape)
        print(dot_product)
        if dot_product > 0:
            y_vec[column] = 1
        elif dot_product < 0:
            y_vec[column] = -1
    
    return y_vec



# Train Images
x = readImg('files/bin_image5.gif', 145)
# array2img(x)
x_vec = mat2vec(x)
# print(len(x_vec))

w = create_W(x_vec)
# print(w.shape)
# print(w[0][:])
# print(w)

# Test Images
y = readImg('files/bin_corr2.gif', 145)
y_img_shape = y.shape

y_vec = mat2vec(y)
y_vec_result = update(w, y_vec, 0.8, 1)
print(y_vec_result)
y_vec_result = y_vec_result.reshape(y_img_shape)
print(y_vec_result)
array2img(y, y_vec_result, x)


