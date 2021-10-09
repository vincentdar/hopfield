"""
Author: Vincent Darmawan
Title: Hopfield Network for denoising Images
Github: https://github.com/vincentdar/hopfield/blob/master/app.py

References:
- https://github.com/takyamamoto/Hopfield-Network
- http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html
- Rekaman kelas

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def array2img(test_data, predict_data):
    y = np.zeros(test_data[0].shape)
    y[test_data[0] == 1] = 255
    y[test_data[0] == -1] = 0

    test_img_0 = Image.fromarray(y)

    y = np.zeros(test_data[0].shape)
    y[test_data[1] == 1] = 255
    y[test_data[1] == -1] = 0

    test_img_1 = Image.fromarray(y)

    y = np.zeros(test_data[0].shape)
    y[test_data[2] == 1] = 255
    y[test_data[2] == -1] = 0

    test_img_2 = Image.fromarray(y)

    y = np.zeros(predict_data[0].shape)
    y[predict_data[0] == 1] = 255
    y[predict_data[0] == -1] = 0

    predict_img_0 = Image.fromarray(y)

    y = np.zeros(predict_data[0].shape)
    y[predict_data[1] == 1] = 255
    y[predict_data[1] == -1] = 0

    predict_img_1 = Image.fromarray(y)

    y = np.zeros(predict_data[0].shape)
    y[predict_data[2] == 1] = 255
    y[predict_data[2] == -1] = 0

    predict_img_2 = Image.fromarray(y)


    figure, axis = plt.subplots(3, 2)
    axis[0][0].imshow(test_img_0)
    axis[0][0].set_title("Test Image")
    axis[0][1].imshow(predict_img_0)
    axis[0][1].set_title("Predict Image")
    axis[1][0].imshow(test_img_1)
    axis[1][0].set_title("Test Image")
    axis[1][1].imshow(predict_img_1)
    axis[1][1].set_title("Predict Image")
    axis[2][0].imshow(test_img_2)
    axis[2][0].set_title("Test Image")
    axis[2][1].imshow(predict_img_2)
    axis[2][1].set_title("Predict Image")
    plt.show()
    

    return None

def update(w, y_vec, theta=0.5, time=100):
    length = len(y_vec)
    columns = np.arange(0, length)
    np.random.shuffle(columns)

    itr = 0
    for i in range(time):
        if itr >= 6300:
            itr = 0
        
        column = columns[itr]
        
        dot_product = np.dot(w[column][:], y_vec) - theta
        
        if dot_product > 0:
            y_vec[column] = 1
        elif dot_product < 0:
            y_vec[column] = -1

        itr += 1
    
    return y_vec


training_files = ['files/bin_image1.gif', 
                'files/bin_image2.gif',
                'files/bin_image3.gif', 
                'files/bin_image4.gif', 
                'files/bin_image5.gif']
test_files = ['files/bin_corr1.gif', 'files/bin_corr2.gif', 'files/bin_corr3.gif']



def hopfield(training_files, test_files, theta=0.5, iteration=50000): 
    x_array = []
    y_array = []
    predicted_array = []
    w = None

    print("Training Weight Matrix")
    for file in training_files:
        print("Training File:", file)
        # Train Images
        x = readImg(file, 145)

        # Add to Array for display
        x_array.append(x)

        x_vec = mat2vec(x)
        if w is None:
            w = create_W(x_vec)
        else:
            w += create_W(x_vec)

    print("Denoising Images")
    print("INFO:", "Theta:", theta, "Iteration:", iteration)
    for file in test_files:
        print("Testing File:", file)
        # Test Images
        y = readImg(file, 145)

        y_array.append(y)

        y_img_shape = y.shape

        y_vec = mat2vec(y)
        y_vec_result = update(w, y_vec, theta, iteration)
        y_vec_result = y_vec_result.reshape(y_img_shape)

        predicted_array.append(y_vec_result)

    array2img(y_array, predicted_array)


if __name__ == '__main__':
    hopfield(training_files, test_files, 0.8, 1000)



