import numpy as np
import cv2
from matplotlib import pyplot as plt
from functions import load_images
from functions import filter_image
from functions import get_borders
from functions import get_border_image
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Load images
original_images, images_names = load_images("*.bmp")
assert len(original_images) > 0
print("found " + str(len(original_images)) + " images" )

for i, image, image_name in zip(range(len(original_images)), original_images, images_names):
    print('processing',i,'image')
    
    image_left  = image[:, :(int(image.shape[1] / 2))]
    image_right = image[:,  (int(image.shape[1] / 2)):]
    
    filtered_image_left =  filter_image(image_left, vetrical_alignment = 10, horizontal_alignment = 10)
    filtered_image_right = filter_image(image_right, vetrical_alignment = 10, horizontal_alignment = 10)
    
    # getting borders on images with help of old algorith (на производную)\n",
    left_l_border_ind,  left_r_border_ind  = get_borders(filtered_image_left, 1)
    right_l_border_ind, right_r_border_ind = get_borders(filtered_image_right, 1)
    
    Additional_size = 32
    # getting images of borders
    image_of_left_1st_border, shift  = get_border_image(filtered_image_left, left_l_border_ind,  additional_size=Additional_size)
    image_of_left_2nd_border, shift  = get_border_image(filtered_image_left, left_r_border_ind,  additional_size=Additional_size)
    image_of_right_1st_border, shift = get_border_image(filtered_image_right, right_l_border_ind, additional_size=Additional_size)
    image_of_right_2nd_border, shift = get_border_image(filtered_image_right, right_r_border_ind, additional_size=Additional_size)
    
    first_gap_data = (image_of_left_1st_border[:, -Additional_size: ] + image_of_left_2nd_border[:, :Additional_size] ) / 2
    second_gap_data = (image_of_right_1st_border[:, -Additional_size: ] + image_of_right_2nd_border[:, :Additional_size] ) / 2
    
    brightness_1 = np.mean(first_gap_data, axis=1)
    brightness_2 = np.mean(second_gap_data, axis=1)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.minorticks_on()
    #  Определяем внешний вид линий вспомогательной сетки:
    ax.grid(which='minor', color = 'k', linestyle = ':')
    ax.grid(which='major', color = 'k', linestyle = '-')
    # ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    plt.plot(brightness_1, label='яркость левой щели')
    plt.plot(brightness_2, label='яркость правой щели')
    plt.hlines(255, 0, len(brightness_2), 'r')
    plt.legend()
    
    plt.ylabel('яркость')
    plt.xlabel('пиксели') # вертикальные пиксели в изображении
    plt.savefig(image_name + '_brightness.png', dpi=800)
    
print('finished')
