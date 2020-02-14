import numpy as np
import cv2
from matplotlib import pyplot as plt
from functions import get_4_borders_from_image_fast
from functions import load_images
from functions import create_simple_approximation
from functions import create_complex_approximation

from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

print('started creating of table...')

# Load images
original_images, images_names = load_images("*.bmp")
assert len(original_images) == 1

original_image = original_images[0].copy()
im_name = images_names[0]


border_1,border_2,border_3,border_4 = get_4_borders_from_image_fast(original_image)
print('found borders, calculating diamentr')
slice_count = 16
slice_size = int(border_1.shape[0] / slice_count)

# x_axis_complex, tablet_1_y_axis_complex = create_complex_approximation(border_2, slice_size, slice_count)
# x_axis_complex, razor_1_y_axis_complex = create_complex_approximation(border_1, slice_size, slice_count)
# delta_data_1 = np.abs(tablet_1_y_axis_complex - razor_1_y_axis_complex)

# x_axis_complex, tablet_2_y_axis_complex = create_complex_approximation(border_3, slice_size, slice_count)
# x_axis_complex, razor_2_y_axis_complex = create_complex_approximation(border_4, slice_size, slice_count)
# delta_data_2 = np.abs(tablet_2_y_axis_complex - razor_2_y_axis_complex)

x_axis, tablet_1_y_axis = create_complex_approximation(border_2, slice_size, slice_count)
x_axis, razor_1_y_axis  = create_simple_approximation(border_1, slice_size)
delta_data_1 = np.abs(tablet_1_y_axis - razor_1_y_axis)

x_axis, tablet_2_y_axis = create_complex_approximation(border_3, slice_size, slice_count)
x_axis, razor_2_y_axis  = create_simple_approximation(border_4, slice_size)
delta_data_2 = np.abs(tablet_2_y_axis - razor_2_y_axis)



factor_mcm = 1.94   # коэффициент перевода в микрометры 

fig, ax = plt.subplots(figsize=(9, 7))
ax.minorticks_on()
ax.grid(which='minor', color = 'k', linestyle = ':')
ax.yaxis.set_minor_locator(AutoMinorLocator(10))
# plt.figure('table')
plt.ylabel('микрометры')
plt.xlabel('микрометры') # вертикальные пиксели в изображении

real_diametr = int(im_name[-8:-4])
print('real diametr = ' + str(real_diametr))

filter_size = 32
kernel = np.ones((filter_size, 1), np.float32) / (filter_size * 1)

table = real_diametr - (10000 - (delta_data_2 + delta_data_1) * factor_mcm)

table_filtered = cv2.filter2D(table,-1,kernel)

np.savetxt('../table.txt', table_filtered, fmt='%f')
plt.grid()
plt.plot(table_filtered)
plt.show()

print('finished')
