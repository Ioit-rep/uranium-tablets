import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import glob
import random
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


# функции демонстрации изображения
def show_image(img, name):
    plt.figure(name)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

# загрузка всех изображений по пути
def load_images(path):
    filenames = [img for img in glob.glob(path)] 
    filenames.sort()

    assert len(filenames) >= 1
    
    images = []
    for img_filename in filenames:
        n = cv2.imread(img_filename)
        n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        n = n.astype('float')
        images.append(n)

    return images, filenames
    
# фильтрация изображения линейным фильтром
def filter_image(image, vetrical_alignment, horizontal_alignment):
    filtered_image = image.copy().astype('double')

    kernel = np.ones((vetrical_alignment,horizontal_alignment), np.float32) / (vetrical_alignment * horizontal_alignment)
    filtered_image = cv2.filter2D(filtered_image,-1,kernel)
    
    return filtered_image

# построение картины производной
def analyze_derivative(image, axis, name):
    img_diff = np.diff(image.astype('float'), axis=axis)
    _max = np.max(img_diff)
    img_diff = np.abs(img_diff * 255 / _max)
    return img_diff

# пролучение ширины пика на полувысоте
def get_half_width_on_half_hight(derivative_data, env_size = 50):
    der_max = np.max(derivative_data)
    der_max_ind = np.argmax(derivative_data)
    
    
    left_ind = max(der_max_ind - 50, 0)
    right_ind = min(der_max_ind + 50, derivative_data.shape[0])
    # environment = derivative_data[left_ind, right_ind]
    
    absolute_val_array = np.abs(derivative_data - der_max / 2)
    
    
    left_half_height_ind  = left_ind    + absolute_val_array[left_ind : der_max_ind].argmin()
    right_half_height_ind = der_max_ind + absolute_val_array[der_max_ind : right_ind].argmin()
    
    width = right_half_height_ind - left_half_height_ind
    
    return der_max_ind, width
    
    