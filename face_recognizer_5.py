import face_recognition
import cv2 
import os
import pickle
print(cv2.__version__)

Encodings=[]
#learnings of the Known Faces

Names=[]
#Names of The Known Faces

j=0

image_dir='/home/abby/Desktop/PyPro/faceRecognizer/demoImages/known'
# the Folder which has all the Images
# to Walk through all the files of the Folder(Known0)
for root,dirs,files in os.walk(image_dir):
    print(files)
    for file in files:
        path=os.path.join(root,file) # Joining the root found in the Image directory with The file name\
        print(path)
        # Getting The name of the person from then file
        name =os.path.splitext(file)[0]
        print(name)
        person=face_recognition.load_image_file(path) # loading the person's Pic to the variable "person"
        encoding=face_recognition.face_encodings(person)[0] # Learning the "Person"'s Face into Variable "encoding"
        Encodings.append(encoding) #appending the learning of the person to the entire list
        Names.append(name)  #appending the Name of the person to the entire list

print(Names)

with open('train.pkl','wb')as f:
        pickle.dump(Names,f)
        pickle.dump(Encodings,f)
Encodings=[]
Names=[]
with open('train.pkl','rb') as f:
        Names=pickle.load(f)
        Encodings=pickle.load(f)     


font=cv2.FONT_HERSHEY_SIMPLEX
image_dir='/home/abby/Desktop/PyPro/faceRecognizer/demoImages/unknown'
for root,dirs,files in os.walk(image_dir):
    for file in files:
        print(root)
        print(file)
        testImagePath=os.path.join(root,file)
        testImage=face_recognition.load_image_file(testImagePath)
        #loading the unknown Image in to "testImage"
        facePositions=face_recognition.face_locations(testImage)
        #locating the faces in the "testImage"
        allEncodings=face_recognition.face_encodings(testImage,facePositions)
        # learning all the face encodings from the "testImage" with aid of face Positions
        testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
        for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
            name="Unknown Person"
            matches=face_recognition.compare_faces(Encodings,face_encoding)
    # Matching All faces with that one face from the "testImage"
            if True in matches:
                first_match_index=matches.index(True)
                name=Names[first_match_index]

            cv2.rectangle(testImage,(left,top),(right,bottom),(0,0,255),2)
               #draws a rectangle across the face
            cv2.putText(testImage,name,(left,top-6),font,.75,(0,255,255),2)
               #Puts a text with name above the rectangle 


        cv2.imshow('myWindow',testImage)
        cv2.moveWindow('myWindow',0,0)

        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()

