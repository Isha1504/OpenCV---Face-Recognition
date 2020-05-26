#Importing modules
import cv2
import numpy as np

#Taking input about user name and the number of snaps we need to store by that name
name=input("Enter the name of the person ")
num_of_imgs=(int)(input("Enter the number of images "))

#Open webcam and load cascade file for face detection
cam=cv2.VideoCapture(0)
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#A list which stores the all the shots taken
mugshots=[]

#Loop goes for the no of images we want to get stored
while num_of_imgs:
    
    #Frame by frame video capture -> ret - boolean and frame - each frame that is returned
    ret,frame=cam.read()
    if not ret:
        continue
    
    #Store faces detected in faces - stores the starting and ending points in the form of x,y,w,h
    faces=classifier.detectMultiScale(frame,1.3,5)
    
    #We sort faces according to the area of the rectangle - w*h i.e (e[2]*e[3]) by descending order so that maximum area gets first 
    faces=sorted(faces, key=lambda e:e[2]*e[3],reverse=True)
    
    #If no faces are detected then we continue the loop
    if not faces:
        continue
    
    #Other wise face is the rectangle with the largest area i.e. 0th element
    faces=[faces[0]]

    #Draw Rectangle
    for face in faces:
        #We detect point and diagonal point to draw a rectangle
        x,y,w,h=face
        cropped_image=frame[y:y+h,x:x+w]
        #Crop image after resizing it
        cropped_image=cv2.resize(cropped_image,(100,100))
        #Appending the cropped image in mugshots
        mugshots.append(cropped_image)
        #Decrease the num of images 
        num_of_imgs=num_of_imgs-1

#We convert the whole series of shots stores into a numpy array
mugshots=np.array(mugshots)

#print(mugshots.shape)

#Saving the mugshots along with name in face dataset folder 
np.save('face_dataset/'+name,mugshots)

#Release webcam and close all the windows
cam.release()
cv2.destroyAllWindows()
