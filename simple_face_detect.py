#Importing Modules
import cv2

#Open webcam
cam=cv2.VideoCapture(0)

#Load Cascade file for face recognization
classifier=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#The main code for face detect
while True:
    
    # Capture frame by frame
    #ret is a boolean regarding whether or not there was a return at all
    #frame is each frame that is returned. 
    #(If there is no frame, you wont get an error, you will get None.)
    ret,frame=cam.read()
    if not ret:
        continue  
    #Recognise each face and stores in faces
    faces=classifier.detectMultiScale(frame,1.3,5)
    # Draw rectangle along the faces
    for face in faces:
        # tuple unpacking
        x,y,w,h=face  
        #x,y is one edge and x+w,y+h is diagonal point where:
            # h- height
            #w- weight
         
        #Draw rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
        # start ,diagnol point,color<BGR>,thickness
    
    #Show image with title as mentioned
    cv2.imshow("Face detect",frame)

    #If we get a key(here is q) we will exit the while loop with a break
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

#Releases the webcam, then closes all the windows.      
cam.release()
cv2.destroyAllWindows()