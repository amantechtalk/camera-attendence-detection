
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import datetime
now = datetime.datetime.now()
now1= now.strftime("%Y-%m-%d %H:%M:%S")
print ("Current date and time : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
#/dlib.shape_predictor("C:/xampp1/htdocs/xampp/cameraattendence/shape_predictor_68_face_landmarks.dat")




def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("C:/xampp1/htdocs/xampp/cameraattendence/faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("C:/xampp1/htdocs/xampp/cameraattendence/faces/" + f)
                 
                    try:
                     encoding = fr.face_encodings(face)[0]
                     encoded[f.split(".")[0]] = encoding
                    except:
                        print("")
 
    return encoded
def data_fill(a,b):
    
             import mysql.connector
             db_connection = mysql.connector.connect(
  	            host="localhost",
  	            user="root",
                passwd="",
                database="mydb1"  
                                           )
             print(db_connection)
             db_cursor = db_connection.cursor()
             
             student_sql_query =( "INSERT INTO myusers(time,name)" "VALUES(%s, %s)")

             #Execute cursor and pass query as well as student data
             data=(a,b)
             #Execute cursor and pass query of employee and data of employee
             db_cursor.execute(student_sql_query,data)
             db_connection.commit()
    
             return 

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("C:/xampp1/htdocs/xampp/cameraattendence/faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:
        cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
 
         #resize the window according to the screen resolution
        cv2.resizeWindow('Resized Window', 400, 400)
 
        cv2.imshow('Resized Window', img)
        
        if cv2.waitKey(50) :
            return face_names 
        cv2.destroyAllWindows()      
           
		
VideoCaptureObject=cv2.VideoCapture(0)
result=True
while(result):
    ret,frame =VideoCaptureObject.read()
    cv2.imshow('aman',frame)
    cv2.waitKey(500)
    cv2.imwrite('C:/xampp1/htdocs/xampp/cameraattendence/test1.jpg',frame)
    result=False
VideoCaptureObject.release()
cv2.destroyAllWindows()    
    
papa=classify_face("C:/xampp1/htdocs/xampp/cameraattendence/test1.jpg")

# Python program to convert a list to string 
    
# Function to convert   
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1  
        
        
# Driver code     

papa1=listToString(papa)
print(papa1)
data_fill(now1,papa1)
cv2.destroyAllWindows()  
   
