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

def array2img_single(x):
    y = np.zeros(x.shape)
    y[x == 1] = 255
    y[x == -1] = 0

    plt.imshow(y)
    plt.show()

def array2img_noise(test_data, predict_data):
    test_image = []
    for img in test_data:
        y = np.zeros(img.shape)
        y[img == 1] = 255
        y[img == -1] = 0

        test_image.append(Image.fromarray(y))

    predict_image = []
    for img in predict_data:
        y = np.zeros(img.shape)
        y[img == 1] = 255
        y[img == -1] = 0

        predict_image.append(Image.fromarray(y))

    # Single Image
    # figure, axis = plt.subplots(1, 2)
    # axis[0].imshow(test_image[0])
    # axis[0].set_title("Test Image")
    # axis[1].imshow(predict_image[0])
    # axis[1].set_title("Predict Image")

    if len(test_image) < 6:
        figure, axis = plt.subplots(5, 2)
        axis[0][0].imshow(test_image[0])
        axis[0][0].set_title("Test Image")
        axis[0][1].imshow(predict_image[0])
        axis[0][1].set_title("Predict Image")
        axis[1][0].imshow(test_image[1])
        axis[1][0].set_title("Test Image")
        axis[1][1].imshow(predict_image[1])
        axis[1][1].set_title("Predict Image")
        axis[2][0].imshow(test_image[2])
        axis[2][0].set_title("Test Image")
        axis[2][1].imshow(predict_image[2])
        axis[2][1].set_title("Predict Image")
        axis[3][0].imshow(test_image[3])
        axis[3][0].set_title("Test Image")
        axis[3][1].imshow(predict_image[3])
        axis[3][1].set_title("Predict Image")
        axis[4][0].imshow(test_image[4])
        axis[4][0].set_title("Test Image")
        axis[4][1].imshow(predict_image[4])
        axis[4][1].set_title("Predict Image")
    else:
        figure, axis = plt.subplots(6, 2)
        axis[0][0].imshow(test_image[0])
        axis[0][0].set_title("Test Image")
        axis[0][1].imshow(predict_image[0])
        axis[0][1].set_title("Predict Image")
        axis[1][0].imshow(test_image[1])
        axis[1][0].set_title("Test Image")
        axis[1][1].imshow(predict_image[1])
        axis[1][1].set_title("Predict Image")
        axis[2][0].imshow(test_image[2])
        axis[2][0].set_title("Test Image")
        axis[2][1].imshow(predict_image[2])
        axis[2][1].set_title("Predict Image")
        axis[3][0].imshow(test_image[3])
        axis[3][0].set_title("Test Image")
        axis[3][1].imshow(predict_image[3])
        axis[3][1].set_title("Predict Image")
        axis[4][0].imshow(test_image[4])
        axis[4][0].set_title("Test Image")
        axis[4][1].imshow(predict_image[4])
        axis[4][1].set_title("Predict Image")
        axis[5][0].imshow(test_image[5])
        axis[5][0].set_title("Test Image")
        axis[5][1].imshow(predict_image[5])
        axis[5][1].set_title("Predict Image")
    plt.show()

    return None

def array2img(test_data, predict_data):
    test_image = []
    for img in test_data:
        y = np.zeros(img.shape)
        y[img == 1] = 255
        y[img == -1] = 0

        test_image.append(Image.fromarray(y))

    predict_image = []
    for img in predict_data:
        y = np.zeros(img.shape)
        y[img == 1] = 255
        y[img == -1] = 0

        predict_image.append(Image.fromarray(y))

    figure, axis = plt.subplots(3, 2)
    axis[0][0].imshow(test_image[0])
    axis[0][0].set_title("Test Image")
    axis[0][1].imshow(predict_image[0])
    axis[0][1].set_title("Predict Image")
    axis[1][0].imshow(test_image[1])
    axis[1][0].set_title("Test Image")
    axis[1][1].imshow(predict_image[1])
    axis[1][1].set_title("Predict Image")
    axis[2][0].imshow(test_image[2])
    axis[2][0].set_title("Test Image")
    axis[2][1].imshow(predict_image[2])
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

# With cat
# training_files = ['files/bin_image1.gif', 
#                 'files/bin_image2.gif',
#                 'files/bin_image3.gif', 
#                 'files/bin_image4.gif', 
#                 'files/bin_image5.gif',
#                 'files/cat-resized.png']
# Without cat
training_files = ['files/bin_image1.gif', 
                'files/bin_image2.gif',
                'files/bin_image3.gif', 
                'files/bin_image4.gif', 
                'files/bin_image5.gif']
# training_files = ['files/cat-resized.png']               
test_files = ['files/bin_corr1.gif', 
            'files/bin_corr2.gif', 
            'files/bin_corr3.gif']



def hopfield(training_files, test_files, theta=0.5, iteration=50000): 
    x_array = []
    y_array = []
    predicted_array = []
    noise_img = []
    w = None

    print("Training Weight Matrix")
    for file in training_files:
        print("Training File:", file)
        # Train Images
        x = readImg(file, 145)

        # Add to Array for display
        x_array.append(x)
        x_noise = addNoise(x)
        noise_img.append(x_noise)

        # Add Noise to Image
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

    y_noise_array = []
    predicted_noise_array = []
    itr = 1
    for img in noise_img:
        print("Noise Image:", itr)
        itr += 1
        # Test Images
        y = img

        y_noise_array.append(y)

        y_img_shape = y.shape

        y_vec = mat2vec(y)
        y_vec_result = update(w, y_vec, theta, iteration)
        y_vec_result = y_vec_result.reshape(y_img_shape)

        predicted_noise_array.append(y_vec_result)

    array2img_noise(y_noise_array, predicted_noise_array)

def addNoise(x):
    noise = np.zeros([x.shape[0], x.shape[1]])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            random = np.random.randint(10)
            if random == 1:
                if x[i][j] > 0:
                    noise[i][j] = -1
                elif x[i][j] < 0:
                    noise[i][j] = 1
            else:
                noise[i][j] = x[i][j]
    
    # array2img_single(noise)

    return noise
    
    

if __name__ == '__main__':
    hopfield(training_files, test_files, 0.8, 100000)



