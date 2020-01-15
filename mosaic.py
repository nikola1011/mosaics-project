import os
import datetime
import photomosaic as pm
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import colorspacious
import sys
#from skimage.io import imsave, imread
#from skimage import data
from skimage import img_as_float
#from skimage.color import gray2rgb
from skimage.transform import resize # may be useful for resize (keep it)
#from skimage.util import crop
from PIL import Image
# This works if this script is a level above the script that being imported
from img_object_detection_tensorflow import detect_images
from progress_bar import print_progress_bar

def concatenateImages(img_files, grid_size=None, tile_size=None):
    '''
    Concatenate images into one single (large) image
    @params:
        img_files - list of image file paths
        grid_size - tuple holding number of tiles by width and height
        tile_size - tuple specifying (width, height) of a tile # all tiles are the same size
    '''

    if (tile_size == None):
        temp_img = Image.open(img_files[0]) # open first tile image just to get its size
        #all tiles are the same size as the first image tile
        tile_width = temp_img.size[0]
        tile_height = temp_img.size[0]

    if (grid_size == None):
        grid_size_eq = math.floor(math.sqrt(len(img_files))) # equally divide tiles by width and height by calculating square root of total number of tiles
        grid_size = (grid_size_eq, grid_size_eq) 
    
    total_width = grid_size[0] * tile_width # in pixels
    total_height = grid_size[1] * tile_height # in pixels

    new_im = Image.new('RGB', (total_width, total_height))
    #14720px WIDTH WORKING!
    #18560px WIDTH WORKING!
    #26560px WIDTH WORKING!

    progressCounter = 0
    totalIterations = grid_size[0] * grid_size[1]

    for y in range(0, total_height, tile_height):
        for x in range(0, total_width, tile_width):
            new_im.paste(Image.open(img_files[progressCounter]), (x, y))
            progressCounter += 1
            print_progress_bar(progressCounter, totalIterations, 'Progress:', 'completed.')

    output_path = 'test_' + str((tile_width, tile_height)) + 'px_' + str(grid_size) + 'gridSize_' + str(datetime.datetime.now()) + '.jpg'
    new_im.save(output_path)
    print (f'Finished. Saved to {output_path}')

######################################################################
def create_mosaic(input_img, images_dataset_path, grid_dims):
        
    # Load a sample image
    image = cv2.imread(img_path)
    image = img_as_float(image) #ensure image is float ranging from 0 to 1

    # Analyze the collection (the "pool") of images.
    pool = pm.make_pool(images_dataset_path)
    
    # Use perceptually uniform colorspace for all analysis.
    converted_img = pm.perceptual(image)

    # Adapt the color palette of the image to resemble the palette of the pool.
    #adapted_img = pm.adapt_to_pool(converted_img, pool)
    adapted_img = converted_img

    #scale = 1
    #scaled_img = Image.new('RGB', (adapted_img.shape[0] * scale, adapted_img.shape[1] * scale))
    #scaled_img = Image.new('RGB', (5040, 5040))
    scaled_img = pm.rescale_commensurate(adapted_img, grid_dims=grid_dims, depth=0)

    tiles = pm.partition(scaled_img, grid_dims=grid_dims, depth=0)

    # Reshape the 3D array (height, width, color_channels) into
    # a 2D array (num_pixels, color_channels) and average over the pixels.
    tile_colors = [np.mean(scaled_img[tile].reshape(-1, 3), 0)
                for tile in tiles]

    # Match a pool image to each tile.
    match = pm.simple_matcher(pool)
    matches = [match(tc) for tc in tile_colors]

    matches_list = [x[0] for x in matches]
    
    # Perform neural network object detection to see what classes are on which images
    detect_images(matches_list)

    # Concatenate list of matches images to a single mosaic image
    concatenateImages(matches_list, grid_dims)

    
###############################################
#canvas = np.zeros_like(scaled_img)  # black canvas

# Draw the mosaic.
#mos = pm.draw_mosaic(canvas, tiles, matches, scale=5)
#mos = draw_mosaic(canvas, tiles, matches, scale=5)

# Save the mosaic
#filename = 'mosaic_' + str(datetime.datetime.now()) + ".png"
#output_path = '/home/nikola/Git/mosaics-code-from-web/' + filename
#imsave(output_path, mos)

# Show the mosaic
#imgplot = plt.imshow(mos)
#plt.show()
###############################################

if __name__ == "__main__":
    # Change working dir to dir of the .py script that is executed 
    # Used for case when running the script in VS Code via 'Run Python file in terminal' command
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working dir {os.getcwd()}")

    img_path = "/home/nikola/Pictures/joker-before-show-720p.jpg"
    #img_path = "/home/nikola/Pictures/joker720p.jpg"
    #img_path = "/home/nikola/Pictures/joker-movie-banner_1920x1080.jpg"
    #img_path = "/home/nikola/Pictures/green-apples.jpg"
    #img_path = "/home/nikola/Pictures/albino-python.jpg"

    #images_dataset_path = '/home/nikola/Git/python-photo-mosaic/frames/*.jpg'
    #images_dataset_path = '/home/nikola/Git/python-photo-mosaic/frames-320x320/*.jpg'
    images_dataset_path = '/home/nikola/Git/python-photo-mosaic/frames-32x32/*.jpg'
    #images_dataset_path = '/home/nikola/Git/mosaics-average-color/32x32/*.jpeg'
    #images_dataset_path = '/home/nikola/Git/mosaics-average-color/320x320_step7/*.jpeg'
    #images_dataset_path = '/home/nikola/Git/mosaics-average-color/24x24/*.jpeg'

    grid_dims = (80, 80)
    
    create_mosaic(img_path, images_dataset_path, grid_dims)
