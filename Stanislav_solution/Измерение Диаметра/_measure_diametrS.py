import numpy as np
import cv2
from matplotlib import pyplot as plt
from functions import get_4_borders_from_image_fast
from functions import load_images
from functions import create_simple_approximation
from functions import create_complex_approximation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# Load images
original_images, images_names = load_images("*.bmp")
assert len(original_images) > 0
print("found " + str(len(original_images)) + " images" )

for i, image, image_name in zip(range(len(original_images)), original_images, images_names):
    print('processing',i,'image')
    
    print('founding borders...')
    border_1,border_2,border_3,border_4 = get_4_borders_from_image_fast(image)

    print('calculating diametr...')
    slice_count = 16
    slice_size = int(border_1.shape[0] / slice_count)

    x_axis_complex, tablet_1_y_axis_complex = create_complex_approximation(border_2, slice_size, slice_count)
    x_axis_complex, razor_1_y_axis_complex = create_complex_approximation(border_1, slice_size, slice_count)
    delta_data_1 = np.abs(tablet_1_y_axis_complex - razor_1_y_axis_complex)

    x_axis_complex, tablet_2_y_axis_complex = create_complex_approximation(border_3, slice_size, slice_count)
    x_axis_complex, razor_2_y_axis_complex = create_complex_approximation(border_4, slice_size, slice_count)
    delta_data_2 = np.abs(tablet_2_y_axis_complex - razor_2_y_axis_complex)

    factor_mcm = 1.94   # коэффициент перевода в микрометры 

    table = np.loadtxt('table.txt', dtype=float)

    data_diametr = (10000 - (delta_data_2 + delta_data_1) * factor_mcm)
    
    filter_size = 32
    kernel = np.ones((filter_size, 1), np.float32) / (filter_size * 1)

    data_diametr_filtered = cv2.filter2D(data_diametr,-1,kernel).reshape(data_diametr.shape[0])
    
    measured_diametr = table + data_diametr_filtered
    
    file_name = image_name + '_diametr.txt'
    
    np.savetxt(file_name, measured_diametr, fmt='%f')

    with open(file_name, 'a') as the_file:
        the_file.write('\nСредний диаметр: ' + str(np.mean(measured_diametr)))

    print('mean diametr = ' + str(np.mean(measured_diametr)))
    
    # plt.figure('Diametr')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.minorticks_on()
    #  Определяем внешний вид линий вспомогательной сетки:
    ax.grid(which='minor', color = 'k', linestyle = ':')
    ax.grid(which='major', color = 'k', linestyle = '-')
    # ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.ylabel('микрометры')
    plt.xlabel('микрометры') # вертикальные пиксели в изображении
    x_data = np.arange(len(measured_diametr)) * factor_mcm
    plt.plot(x_data, measured_diametr)
    plt.hlines(np.mean(measured_diametr), 0, x_data[-1], 'r')
    plt.savefig(image_name + '_diametr.png', dpi=800)
    
print('finished')
