from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import inspect
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

with tf.Graph().as_default():
      
        with tf.Session() as sess:

            # Get the paths for the corresponding images
            folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
            extracted_dict_output_path = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'extracted_dict.pickle')
            paths = []
            for sub_folder, subdir, folder_list in os.walk(folder):
                for file in folder_list:
                    paths.append(os.path.join(sub_folder, file))
            #

            #np.save("images.npy",paths)
            # Load the model
            model_directory = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '20180402-114759')
            # facenet.load_model(r"C:\Users\AJain7\Desktop\arpit's_model\model\20180402-114759")
            facenet.load_model(model_directory)

            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = 160
            embedding_size = embeddings.get_shape()[1]
            extracted_dict = {}
            
            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):
                # print(i, filename)
                images = facenet.load_image(filename, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[filename] =  feature_vector
                if(i%100 == 0):
                    print("completed",i," images")
            
            with open(extracted_dict_output_path,'wb') as f:
                pickle.dump(extracted_dict,f)
            # print('Done dumping')
