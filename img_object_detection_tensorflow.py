import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from utils import label_map_util
#from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from progress_bar import print_progress_bar


def detect_images(img_paths, save_detected_images=False, detection_threshold=0.5):

    # Define the video stream
    #cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
    print (f'TensorFlow version {tf.__version__}')

    # What model to download.
    # Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    # {model name for downloading} {model name} {speed in ms} {detection in COCO measurement units}
    #MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17' # ssd_inception_v2_coco 42ms 24COCO mAP
    #MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' # slower than ssd_inception_v2_coco_2017_11_17 model, same detection #ssd_resnet_50_fpn_coco ☆76ms 	35 COCO mAP
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' # fastest # same detection as ssd_inception_v2_coco_2017_11_17 #ssdlite_mobilenet_v2_coco 27ms	22 COCO mAP[^1]
    #MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # faster_rcnn_nas 1833ms 43 COCO mAP # DOES NOT WORK, it gets killed for some unknown reason

    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    path_to_research_folder = "/home/nikola/Git/models/research/object_detection/data/"
    PATH_TO_LABELS = os.path.join('data', path_to_research_folder + 'mscoco_label_map.pbtxt')


    # Number of classes to detect
    NUM_CLASSES = 90

    # Download Model
    if not os.path.exists(MODEL_FILE):
        print (f"Downloading {MODEL_NAME} model...")
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
    else:
        print(f"Model {MODEL_NAME} already downloaded")

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    # Helper code
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def load_images(img_paths):
        ''' Load images via generator for less memory usage '''
        
        for img_path in img_paths:
            if not os.path.exists(img_path):
                print(f"File could not be found. Check path and file extension. Entered path is {img_path}")
                exit(0)

            if not os.path.isfile(img_path):
                print(f"File is not a valid file. Check path and file extension. Entered path is {img_path}")
                exit(0)

            #width, height =  img.size[0], img.size[1]
            #print('Frame size: width, height:', width, height)
            yield Image.open(img_path)

    # Detection
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for counter, img in enumerate(load_images(img_paths), 1):

                if img is None:
                    print ("Image is None")
                    exit(0)

                image_np = load_image_into_numpy_array(img)
                #image_np = load_image_into_numpy_array(image_np)
                #cv2.imshow('Loaded image', image_np)
                #cv2.waitKey(0)
                
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    min_score_thresh=.5)

                # Print detected classes (above threshold level) # TODO: Count the same classes 
                class_names = [category_index[int(i)]['name'] for i in classes[0]]
                above_threshold_scores = [x for x in scores[0] if x > detection_threshold]
                print(f"Detected classes: {list(zip(class_names, above_threshold_scores))}")

                img_filename_with_ext = img.filename.split('/')[-1]
                filename, file_ext = img_filename_with_ext.split('.')[0], img.format
                
                # Print current progress
                print_progress_bar(counter, len(img_paths), prefix=f'Detecting image {img_filename_with_ext}')
                
                # Display output
                #cv2.imshow(f"{img_filename_with_ext} (press 'q' to exit)", cv2.resize(image_np, (800, 600)))
                
                # Save output
                if save_detected_images:
                    img_save_path = str(filename + '_detected_output(' + str(counter) + ').' + file_ext)
                    print(f'Saving detected output image to {img_save_path}')
                    ret = cv2.imwrite(img_save_path, image_np)

                    if ret == False:
                        print(f'Warning. imwrite returned: {ret}')

if __name__ == "__main__":
    # Change working dir to dir of the .py script that is executed 
    # Used for case when running the script in VS Code via 'Run Python file in terminal' command
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working dir {os.getcwd()}")

    img_path = "/home/nikola/Pictures/green-apples.jpg"
    #img_path = "/home/nikola/Pictures/0005.jpg"
    img_paths = [img_path] * 1 #* 3600

    start_time = time.time()
    detect_images(img_paths)
    print("Execution took --- %s seconds ---" % (time.time() - start_time))
    