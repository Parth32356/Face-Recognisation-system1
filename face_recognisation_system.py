# Requirements of Datasets
# creating folder ---> 100 images per folder

# Kaggle is used
import numpy as np
import cv2
import os

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels = []
actor_faces = []
# actors_name = []
# for i in os.listdir("images"):
#     actors_name.append(i)

actors_name = ['Angelina Jolie','Brad Pitt']
# actors_name = ['Angelina Jolie','Brad Pitt','Jennifer Lawrence','Johnny Depp','Megan Fox','Robert Downey Jr','Sandra Bullock','Tom Cruise','Tom Hanks','Will Smith']


path = r"C:\Users\lenovo\Downloads\archive (3)\Celebrity Faces Dataset"
    
for actors in actors_name:
    actors_folder = os.path.join(path,actors)
    print(actors_folder)
    
    for images in os.listdir(actors_folder):
        # print(images)
        actor_img_path = os.path.join(actors_folder,images)
        actor_index = actors_name.append()
        # print(actor_img_path)
        array_img = cv2.imread(actor_img_path)
        gray_img = cv2.cvtColor(array_img,cv2.COLOR_BGR2GRAY)
        
        face_roi = face_detector.detectMultiScale(array_img,1.2,3)
        
        for x,y,w,h in face_roi:
            # cv2.rectangle(array_img,(x,y),(x+w,y+h),(10,255,23),3)
            crop_face = gray_img[y : y+h, x : x+w]
            
            labels.append(actor_index)
            actor_faces.append(crop_face)
            
        
        # cv2.imshow("img",array_img)
        # cv2.waitKey(0)
        
lables_array = np.array(labels)
actor_faces_array = np.array(actor_faces,dtype = "object")
        
# Model ---> algo
        
model = cv2.face.LBPHFaceRecognizer_create()
        
model.train(actor_faces_array,lables_array)
        
model.save("face_recognisation_system.yml")
        
        
    
    
  


