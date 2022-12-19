from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,logout,login as auth_login
from django.views.decorators.cache import cache_page
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_protect
from keras.models import Sequential,load_model
import numpy as np 
import cv2
from imutils.video import VideoStream
import imutils
import cv2,time
import os
import urllib.request
import numpy as np
from django.conf import settings

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["seizure", "NoSeizure"]
n = 20
count = 1
LRCN_model=load_model('./models/LRCN_Approach.h5')

# Create your views here.
def home(request):
    return render(request,'home.html')

def login(request):
     if request.method=="POST":
        username = request.POST.get('user','')
        password = request.POST.get('pass','')
        print(username,password)
        user = authenticate(request,username=username, password=password)
        context = {'user' : username }

        if user is not None:
            auth_login(request,user)
            return render(request,'upload.html',context)

          # A backend authenticated the credentials
        else:
            userName = request.POST.get('username', '')
            email = request.POST.get('email', '')
            passWord = request.POST.get('password', '')
            user = User.objects.create_user(username = userName, email = email, password=passWord)
            user.save()
            return render(request ,'login.html')
            # No backend authenticated the credentials
    
     
     return render(request,'login.html')
    

def logout_view(request):
    logout(request)
    return redirect('home')
        

def upload(request):
    return render(request,'upload.html')

def result(request):
    return render(request,'result.html')

def predictVid(request):
    fileObj= request.FILES['vid']
    fs = FileSystemStorage()
    videoPath = fs.save(fileObj.name,fileObj)
    print(fileObj)
    videoPat = fs.url(videoPath)
    # testVid ='.'+videoPath
    print(videoPath)
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture('./static/images/'+videoPath)

    print(video_reader)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(original_video_height)
    print(original_video_width)
    frames_list = []
    predicted_class_name = ''

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FPS))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
 
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
                     print(cv2.CAP_PROP_POS_FRAMES)
                     video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        
        # Read a frame.
                     success, frame = video_reader.read()
                     
        # Check if frame is not read properly then break the loop.
                     if not success:
                       break

        # Resize the Frame to fixed Dimensions.
                     resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                     normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
                     frames_list.append(normalized_frame)



    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    print(predicted_label)
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        
    # Release the VideoCapture object. 
    video_reader.release()
    
    
    context = {'videoPat':videoPat, 'action': predicted_class_name, 'confidence': predicted_labels_probabilities[predicted_label]}
    

    return render(request,'result.html',context)




    