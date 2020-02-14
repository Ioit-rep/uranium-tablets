import numpy as np
from functions import get_4_borders_from_image_fast
from functions import load_images
from functions import analyze_gap

from functions import create_simple_approximation
from functions import create_complex_approximation


# Load images
original_images, images_names = load_images("*.bmp")
assert len(original_images) == 1
print("found " + str(len(original_images)) + " images" )
print('processing image...')

number = 0

original_image = original_images[number].copy()
im_name = images_names[number]


border_1,border_2,border_3,border_4 = get_4_borders_from_image_fast(original_image)

print('showing 1st plot...')
analyze_gap(border_razor=border_1, border_tablet=border_2, name=im_name[:-4] + '_left', show_plot=True)

print('showing 2nd plot...')
analyze_gap(border_razor=border_4, border_tablet=border_3, name=im_name[:-4] + '_right', show_plot=True)

print('finished')
