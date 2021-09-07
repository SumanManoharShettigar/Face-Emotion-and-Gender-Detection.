from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import cvlib as cv
                    
# load model
model = load_model('gender_detection.model')
#Cascading classifiers are trained with several hundred "positive" sample views of a particular object and arbitrary "negative" images of the same size
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('model.h5')



# open webcam
webcam = cv2.VideoCapture(0) #0->primary camera
#prediction labels
classes = ['man','woman']
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read() #status to check whether being recorded or not

    # apply face detection
    face, confidence = cv.detect_face(frame)#to detect face we use detect_face from cvlib module


    # loop through detected faces
    for idx, f in enumerate(face):
        
        #emotion empty label
        labels = []
        
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)#2->thickness

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
        
        

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        
        
        
        #emotion test
        
        roi_gray = cv2.cvtColor(face_crop,cv2.COLOR_BGR2GRAY)
        
        
        #trained model is in 48x48 , so we resize. 
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # Resizing an image needs a way to calculate pixel values for the new image from the original one. SO interpolating
        # INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to theINTER_NEAREST method.
        
        
        roi = roi_gray.astype('float')/255.0 #standardise roi
        roi = img_to_array(roi)# convert to array because model is trained on arrays
        roi = np.expand_dims(roi,axis=0)#adds an dimension for batch sizes.eg (2)->(1,2)

        prediction = classifier.predict(roi)[0]#using model, it predicts emotion in ROI
        labela=emotion_labels[prediction.argmax()]# find emotion which has got highest probability, ragmax gives index and it is matched in labels
        
        
        #Emotion End
        
        #gender start
        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))#gender model is trained in 96x96. so resize
        face_crop = face_crop.astype("float") / 255.0 #standardise face_crop
        face_crop = img_to_array(face_crop) # convert to array because model is trained on arrays
        face_crop = np.expand_dims(face_crop, axis=0) #adds an dimension for batch sizes.eg (2)->(1,2)
        
        
        

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)# find gender which has got highest probability, ragmax gives index and it is matched in classes
        label = classes[idx]#finding the prediction using index
        
        #store prediction of gender for display
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)#conf[idx] is percentage
        
        
        #gender end
        
        
        # just coordinates to print predictions
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        
        #write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.putText(frame, labela, (startX, endY+20),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        
        # #Emotion Starts here
        # labels = []
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # faces = face_classifier.detectMultiScale(gray) #if face is detected a rectangle is drawn over it and its coordinates are sent

        # for (x,y,w,h) in faces:#for each face in frame
            
        #     roi_gray = gray[y:y+h,x:x+w]#region of interest only select the face in grayscale
            
        #     # #trained model is in 48x48 , so we resize. 
        #     # roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        #     # # Resizing an image needs a way to calculate pixel values for the new image from the original one. SO interpolating
        #     # # INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to theINTER_NEAREST method.
    
    
    
        #     if np.sum([roi_gray])!=0:#if theres a face
        #         roi = roi_gray.astype('float')/255.0 #standardise roi
        #         roi = img_to_array(roi)# convert to array because model is trained on arrays
        #         roi = np.expand_dims(roi,axis=0)#adds an dimension for batch sizes.eg (2)->(1,2)
    
        #         prediction = classifier.predict(roi)[0]#using model, it predicts emotion in ROI
        #         labela=emotion_labels[prediction.argmax()]# find emotion which has got highest probability, ragmax gives index and it is matched in labels
        #         label_position1 = (x,y-10)
        #         label_position2 = (x,y+h+20)
                
        #         cv2.putText(frame,labela,label_position1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        #         cv2.putText(frame, label, label_position2,  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        #     else:
        #         cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        
        
        
        
        

        # write label and confidence above face rectangle
       #cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        
        
        

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()