#!flask/bin/python
################################################################################################################################
#------------------------------------------------------------------------------------------------------------------------------                                                                                                                             
# This file implements the REST layer. It uses flask micro framework for server implementation. Calls from front end reaches 
# here as json and being branched out to each projects. Basic level of validation is also being done in this file. #                                                                                                                                  	       
#-------------------------------------------------------------------------------------------------------------------------------                                                                                                                              
################################################################################################################################
from flask import Flask, jsonify, abort, request, make_response, url_for,redirect, render_template
#from flask.ext.httpauth import HTTPBasicAuth
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os
import inspect
import sys
import random
from tensorflow.python.platform import gfile
from six import iteritems
sys.path.append('..')
import numpy as np
#from lib.src import retrieve
import retrieve
#from lib.src.align import detect_face
import detect_face
import tensorflow as tf
import pickle
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.platform import gfile
app = Flask(__name__, static_url_path = "")

auth = HTTPBasicAuth()


#==============================================================================================================================
#                                                                                                                              
#    Loading the stored face embedding vectors for image retrieval                                                                 
#                                                                          						        
#                                                                                                                              
#==============================================================================================================================
#with open('../lib/src/face_embeddings.pickle','rb') as f:
#					    	feature_array = pickle.load(f) 

with open(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'extracted_dict.pickle'),'rb') as f:
					    	feature_array = pickle.load(f) 

model_path = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), '20180402-114759')
# model_exp = r"C:\Users\AJain7\Desktop\arpit's_model\model\20180402-114759\model-20180402-114759.meta"
model_exp = os.path.join(model_path, 'model-20180402-114759.meta')
# model_exp_1 = r"C:\Users\AJain7\Desktop\arpit's_model\model\20180402-114759\model-20180402-114759.ckpt-275"
model_exp_1 = os.path.join(model_path, 'model-20180402-114759.ckpt-275')
graph_fr = tf.Graph()
sess_fr = tf.Session(graph=graph_fr)
with graph_fr.as_default():
  saverf = tf.train.import_meta_graph(model_exp)
  saverf.restore(sess_fr, model_exp_1)
  mtcnn_path = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'mtcnn')
  pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, mtcnn_path)
#==============================================================================================================================
#                                                                                                                              
#  This function is used to do the face recognition from video camera                                                          
#                                                                                                 
#                                                                                                                              
#==============================================================================================================================
@app.route('/facerecognitionLive', methods=['GET', 'POST'])
def face_det():
    
    retrieve.recognize_face(sess_fr,pnet, rnet, onet,feature_array)

#==============================================================================================================================
#                                                                                                                              
#                                           Main function                                                        	            #						     									       
#  				                                                                                                
#==============================================================================================================================
@app.route("/")
def main():
    
    return render_template("main.html")
    #return '''<form action="/check" method="get">
  #URL: <input type="text" name="fname"><br>
  #<input type="submit" value="Submit">
#</form>'''

if __name__ == '__main__':
    app.run(debug = True, host= '127.0.0.1')
