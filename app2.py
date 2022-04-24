from __future__ import division, print_function
# coding=utf-8
import os
import glob
import numpy as np
import librosa as lb

# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras import models

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

file_model = r'C:\Users\suyash\Desktop\KACHRA\laohub\Smile in Pain\Ajgar Ke Jalve\Artificiall Intelligence\Neural Networks\Supervised Learning\Recurrent Nets\RNN\Audio\H5\final1000.h5'

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'
# IMG_SIZE = 512

# # Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


# def crop_image_from_gray(img,tol=7):
#     if img.ndim ==2:
#         mask = img>tol
#         return img[np.ix_(mask.any(1),mask.any(0))]
#     elif img.ndim==3:
#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         mask = gray_img>tol
        
#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
#         if (check_shape == 0): # image is too dark so that we crop out everything,
#             return img # return original image
#         else:
#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
#     #         print(img1.shape,img2.shape,img3.shape)
#             img = np.stack([img1,img2,img3],axis=-1)
#     #         print(img.shape)
#         return img

# def circle_crop(img, sigmaX = 30):   
#     """
#     Create circular crop around image centre    
#     """    
#     img = crop_image_from_gray(img)    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     height, width, depth = img.shape    
    
#     x = int(width/2)
#     y = int(height/2)
#     r = np.amin((x,y))
    
#     circle_img = np.zeros((height, width), np.uint8)
#     cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
#     img = cv2.bitwise_and(img, img, mask=circle_img)
#     img = crop_image_from_gray(img)
#     img=cv2.resize(cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128), (320, 320))
#     return img 

# # def preprocess_image(file):
# #     input_filepath = os.path.join('./','test_images_resized','{}.png'.format(file))
# #     output_filepath = os.path.join('./','test_images_resized_preprocessed','{}.png'.format(file))
    
# #     img = cv2.imread(input_filepath)
# #     img = circle_crop(img) 
# #     cv2.imwrite(output_filepath, cv2.resize(img, (IMG_SIZE,IMG_SIZE)))

# def model_predict(img_path):
#     main = r".\model_main_training.h5"
#     post = r".\model_post_training.h5"

#     model = load_model(main, compile=False)
#     end = load_model(post, compile=False)
#     img = cv2.imread(img_path)
#     img = circle_crop(img)
#     img = img/255

#     preds = model.predict(img.reshape(1, 320, 320, 3))
#     preds[1] = np.argmax(preds[1])
#     preds[2] = np.argmax(preds[2])
#     preds[0] = preds[0][0]
#     final = end.predict(tf.keras.backend.constant(np.array(preds).reshape(1, 3)))
#     print(preds, final)
#     # x = preprocess_image(img_path)
#     if final<0.5:
#         out = "No DR"
#     elif final<1.5:
#         out ="Mild"
#     elif final<2.5:
#         out ="Moderate"
#     elif final<3.5:
#         out ="Severe"
#     else:
#         out = "Proliferative DR"
#     # preds = model.predict(x)
#     return out
def preprocessing(x):
    shape = np.array(x).shape
    l = (shape[1] - shape[1]%4)/4
    new = x[:,:int(l*4)]
    # print(x.shape)
    new = np.reshape(new, (64, int(l)))
    new = sequence.pad_sequences(new, maxlen = 120, padding="post")
    new = np.reshape(new, (120, 64))
    new = (new+464.4627)/666.9084
    return new

def model_predict(file_path):
    y, sr = lb.load(file_path)
    mfccs_features = lb.feature.mfcc(y=y, sr=sr, n_mfcc=16)
    x = preprocessing(mfccs_features)
    model = models.load_model(file_model)
    result = model.predict(x.reshape(1,120, 64))
    return result
    

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        
        if preds[0][0]>=0.5 :
            return {'message':"You Have cough and symptoms \n ",'value':str(preds[0][0])}
        else:
            return {'message':"You are save you don't have symptops for covid 19 \n",'value':str(preds[0][0])}
        return str(preds[0][0])
    return None


if __name__ == '__main__':
    app.run(debug=True)

