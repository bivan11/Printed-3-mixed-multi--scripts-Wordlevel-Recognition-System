#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import imutils
#from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


# In[2]:


image = cv2.imread("final1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)


# In[3]:


#ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


# In[4]:


rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))


# In[5]:


#dilate = cv2.dilate(thresh1, None, iterations=1)
dilate = cv2.dilate(thresh1, rect_kern, iterations = 1)


# In[6]:


cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]
#cnts = cnts[0] if len(cnts) ==2 else cnts[1]


# In[7]:


sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[0] * image.shape[0] )
#cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
#cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   


# In[8]:


# sort contours left-to-right
#sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])


# In[ ]:





# In[9]:


orig = image.copy()
i = 0
#rois=[]
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 100):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)
    
    if (w >3 and h > 5):
    
        # Taking ROI of the cotour
        roi = image[y:y+h, x:x+w]
        #print(type(roi))

        # Mark them on the image if you want
        cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)

        # Save your contours or characters
        cv2.imwrite("Images/roi" + str(i) + ".png", roi)

        i = i + 1 


# In[10]:


cv2.imwrite("box1.jpg",orig)


# In[11]:


import numpy as np
# from keras.models import model_from_json
# from keras.models import load_model

# def prediction(orig):
#     # load json and create model
#     json_file = open('dcr1.json', 'r')
    
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
    
#     # load weights into new model
#     loaded_model.load_weights("dcr1.h5")
#     #print("Loaded model from disk")
    
#     loaded_model.save('dcr1.hdf5')
#     loaded_model=load_model('dcr1.hdf5')
    
#     categories  = {'0':'0', 'A':'A', 'B':'B', 'C':"C", 'D':'D', 'E':'E', 'F':'F', 'G':'G', 'H':'H', 'I':'I', 'J':'J', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'O':'O', 'P':'P', 'Q':'Q', 'R':'R', 'S':'S', 'T':'T', 'U':'U', 'V':'V', 'W':'W', 'X':'X', 'Y':'Y','Z':'Z', 'RA':'रा', 'a_':'a', 'b_':'b', 'c_':'c', 'd_':'d', 'e_':'e', 'f_':'f', 'g_':'g', 'h_':'h', 'i_':'i', 'j_':'j', 'k_':'k', 'l_':'l', 'm_':'m', 'n_':'n', 'o_':'o', 'p_':'p', 'q_':'q', 'r_':'r', 's_':'s', 't_':'t', 'u_':'u', 'v_':'v', 'w_':'w', 'x_':'x', 'y_':'y', 'z_':'z', 'kk':'क'} 
#     #categories = categories.split(',')
    
#     x = np.asarray(img, dtype = np.float32).reshape(1, 64, 64, 1)/255
    
#     output = loaded_model.predict(x)
#     output = output.reshape(55)
#     predicted = np.argmax(output)
#     char_label = categories[list(categories.keys())[predicted]]
#     success = output[predicted] * 100
# #     
#     #print(str(char_label[0]),orig)
#     #return char_label, success
# prediction(orig)


# # In[ ]:


# from glob import glob
# import os
# import cv2

# in_dir = 'Images'
    
# infiles = in_dir + '/*.png'
# img_names = glob(infiles)
#print(img_names)


# def classifier(img_names):
# #     in_dir = 'Images'
    
# #     infiles = in_dir + '/*.png'
# #     img_names = glob(infiles)
# #     print(img_names)
#     pred_lbl = ""
#     acc = []
#     i =0
#     for fn in (img_names):
#         #print('processing %s...' % fn)
#         print('string:',fn)
# #         print(type(fn))
# #         fn = np.fromstring(fn, dtype =np.float64)
# #         print(type(r))
#         fn = img_to_array(fn)
#         #print(type(fn))
#         #fn = list(map(float,fn.split('.')))
#         fn = cv2.resize(fn, (64,64))
#     #roi = img_to_array(roi)
#     #roi = preprocess_input(roi)
#     #segment = cv2.GaussianBlur(segment, (3, 3), 0)
#     #segment = cv2.erode(segment, (3, 3), 1)
#     #show(segment)

#         lbl, a = prediction(fn)
#         pred_lbl+=lbl
#         acc.append(a)
#     return pred_lbl, np.array(acc).mean()
# classifier(img_names[0])


# In[18]:


# import os
# from keras.preprocessing import image

# def classifier(folder_path):
#     pred_lbl = ""
#     acc = []
#     folder_path = 'C:/Users/bivan/Desktop/preprocessing/temp/'
#     img_width, img_height = 64, 64
#     for img in os.listdir(folder_path):
#         img = os.path.join(folder_path, img)
#         img = image.load_img(img, target_size=(img_width, img_height))
#         img = image.img_to_array(img)
#         img_name = cv2.resize(img, (64,64))
#         #segment = cv2.GaussianBlur(segment, (3, 3), 0)
#         #segment = cv2.erode(segment, (3, 3), 1)
#         #show(segment)
        
#         lbl, a = prediction(img)
#         pred_lbl+=lbl
#         acc.append(a)
#     return pred_lbl, np.array(acc).mean()
# classifier(folder_path[0])


# In[ ]:





# In[ ]:





# In[17]:


# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt
# #from google.colab.patches import cv2_imshow
# '''from google.colab import files
# upload = files.upload()'''
# model = tf.keras.models.load_model('cnn.model')

# img = cv2.imread(r'C:\Users\bivan\Desktop\preprocessing\Images\roi5.png')
# plt.imshow(img)
# plt.show()
# img_copy = img.copy()

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (64,64))

# img_copy = cv2.GaussianBlur(img_copy, (3,3), 0)
# img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
# _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

# img_final = cv2.resize(img_thresh, (64,64))
# img_final =np.reshape(img_final, (1,64,64,-1))
# plt.imshow(img_thresh,cmap='gray')
# plt.show()

# # img_final = cv2.resize(img, (64,64))
# # img_final =np.reshape(img_thresh,(1,64,64,-1))

# categories  = {'0':'0', 'A':'A', 'B':'B', 'C':"C", 'D':'D', 'E':'E', 'F':'F', 'G':'G', 'H':'H', 'I':'I', 'J':'J', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'O':'O', 'P':'P', 'Q':'Q', 'R':'R', 'S':'S', 'T':'T', 'U':'U', 'V':'V', 'W':'W', 'X':'X', 'Y':'Y','Z':'Z', 'RA':'रा', 'a_':'a', 'b_':'b', 'c_':'c', 'd_':'d', 'e_':'e', 'f_':'f', 'g_':'g', 'h_':'h', 'i_':'i', 'j_':'j', 'k_':'k', 'l_':'l', 'm_':'m', 'n_':'n', 'o_':'o', 'p_':'p', 'q_':'q', 'r_':'r', 's_':'s', 't_':'t', 'u_':'u', 'v_':'v', 'w_':'w', 'x_':'x', 'y_':'y', 'z_':'z', 'kk':'क'} 
#     #categories = categories.split(',')

# #img_pred = categories[np.argmax(model.predict(img_final))]

# predictions = model.predict(img_final)
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely is {} with a {:.2f} percent confidence.".format(categories[list(categories.keys())[np.argmax(score)]], 100*np.max(score))
# )


# In[49]:


import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

# image folder
folder_path = 'C:/Users/bivan/Desktop/preprocessing/Images/'

img_width, img_height = 64, 64

# load the trained model
model = tf.keras.models.load_model('cnn.model')
#model = load_model(model_path)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# load all images into a list
images = []
img_list=[]
categories  = {'0':'0', 'A':'A', 'B':'B', 'C':"C", 'D':'D', 'E':'E', 'F':'F', 'G':'G', 'H':'H', 'I':'I', 'J':'J', 'K':'K', 'L':'L', 'M':'M', 'N':'N', 'O':'O', 'P':'P', 'Q':'Q', 'R':'R', 'S':'S', 'T':'T', 'U':'U', 'V':'V', 'W':'W', 'X':'X', 'Y':'Y','Z':'Z', 'रा':'रा ', 'a_':'a', 'b_':'b', 'c_':'c', 'd_':'d', 'e_':'e', 'f_':'f', 'g_':'g', 'h_':'h', 'i_':'i', 'j_':'j', 'k_':'k', 'l_':'l', 'm_':'m', 'n_':'n', 'o_':'o', 'p_':'p', 'q_':'q', 'r_':'r', 's_':'s', 't_':'t', 'u_':'u', 'v_':'v', 'w_':'w', 'x_':'x', 'y_':'y', 'z_':'z', '1':'क'} 
  
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_copy = cv2.GaussianBlur(img_copy, (3,3), 0)
    #img_copy = cv2.erode(img_copy, (3, 3), 1)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    img_final = np.resize(img_thresh, (64,64))
    img_final =np.reshape(img_final, (1,64,64,-1))
#     plt.imshow(img_thresh, cmap='gray')
#     plt.show()
    #images.append(img_final)
    #print(images)
    #img_list.append(img_thresh)
    predictions = model.predict(img_final)
    score = tf.nn.softmax(predictions[0])
    
    
    result =  [categories[list(categories.keys())[np.argmax(score)]]]
    #images.append(result)
    #print(type(result))
    #print(result)
    for sublist in result:
        #for item in sublist:
        images.append(sublist)
        #print(images)
                                                     
# some_list=['C:/Users/bivan/Desktop/preprocessing/temp/']
# # for img_path in os.listdir(folder_path):
# #     some_list.append(img_path)
    
#print(some_list)

# from glob import glob
# import os
# import cv2

# in_dir = 'temp'
    
# infiles = in_dir + '/*.jpg'
# img_names = glob(infiles)

# # for img_path in os.listdir(folder_path):
# #     img_names.append(img_path)
    
# #print(img_names)
# img_names.append(img_names)


# In[56]:


print("{}{}{}{}{}{}".format(*images))


# In[33]:




# In[12]:


# def load_and_prep_image(filename, img_shape=64):

# #   Reads an image from filename, turns it into a tensor
# #   and reshapes it to (img_shape, img_shape, colour_channel).
  
#   # Read in target file (an image)
#     img = tf.io.read_file(filename)

# #   Decode the read file into a tensor & ensure 3 colour channels 
# #   (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
#     img = tf.image.decode_image(img, channels=1)

# #   Resize the image (to the same size our model was trained on)
#     img = tf.image.resize(img, size = [img_shape, img_shape])

# #   Rescale the image (get all values between 0 and 1)
#     img = img/255.0
#     return img

# letters = []
# for img_name in images:
#      # Import the target image and preprocess it
#     #img =load_and_prep_image(img_name )
#     print(img_name)
#   # Make a prediction
#     #pred = model.predict(tf.expand_dims(img, axis=0))
#     pred = model.predict(img_name)
#     class_names=['A','s', 's', 'i', 'g', 'n', 'm', 'e', 't']

#   # Get the predicted class
#     pred_class = categories[list(categories.keys())[int(tf.round(pred)[0][0])]

#   #save the letter
#     letters.append(pred_class )

# output = []

# for letter in letters:
#     output += letter 
# print(output)


# # In[ ]:


# letters = []
# for some in some_list:
#      # Import the target image and preprocess it
#     img =load_and_prep_image(some)

#   # Make a prediction
#     pred = model.predict(tf.expand_dims(img, axis=0))

#   # Get the predicted class
#     pred_class = class_names[int(tf.round(pred)[0][0])]

#   #save the letter
#     letters.append(pred_class )

# output = []

# for letter in letters:
#     output =+ letter 
# print(output)


# # In[ ]:




