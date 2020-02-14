import numpy as np
import cv2
from matplotlib import pyplot as plt
from functions import load_images
from functions import filter_image
from functions import analyze_derivative
from functions import get_half_width_on_half_hight

# Load images
original_images, images_names = load_images("*.bmp")
assert len(original_images) > 0
print("found " + str(len(original_images)) + " images" )

for i, image, image_name in zip(range(len(original_images)), original_images, images_names):
    print('processing',i,'image')
    
    fil_im = filter_image(image, vetrical_alignment = 30, horizontal_alignment = 30)
    img_der = analyze_derivative(fil_im, axis=1, name=image_name[:-4] + "_derivative")
    
    temp_slice = img_der[1000, :].copy()
    
    widths = np.zeros(4)
    
    env_size = 50
    for i in range(4):
        ind_max, widths[i] = get_half_width_on_half_hight(temp_slice, env_size)
        temp_slice = np.concatenate((temp_slice[:int(ind_max - env_size)], temp_slice[int(ind_max + env_size):]))
    
    
    
    plt.figure(image_name)
    plt.ylim(0, 255)
    plt.plot(img_der[1000, :])
    
    plt.grid()
    
    plt.ylabel('значение производной')
    plt.xlabel('пиксели по горизонтали') # вертикальные пиксели в изображении
    plt.savefig(image_name + '_sharpness_' + str(widths) + '.png', dpi=800)
    
print('finished')
