import cv2
import os
face_detector = cv2.CascadeClassifier('C:\\Users\\Selman\\Desktop\\DL\\face-recognition-cnn\\FacialRecognition\\haarcascade_frontalface_default.xml')

class Dataset:
    def __init__(self,cam):
        self.cam = cam
        self.limit = 30 # dataset picture limit for every face
    def dataset(self,face_id,no_user,count):
        self.face_id = face_id
        ret, img = self.cam.read()
        img = cv2.flip(img, 1)
        if count<=self.limit:
            img = cv2.putText(img,f"Pictures Taken: {count}", (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,0),1, cv2.LINE_AA)
        else:
            img = cv2.putText(img,"Done!", (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0),1, cv2.LINE_AA)
            no_user =False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(gray,
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20))
        if not os.path.exists('photos/'+self.face_id):
                os.mkdir('photos/'+self.face_id)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            if count<self.limit:
                cv2.imwrite(f"photos/{self.face_id}/User." + str(self.face_id) + '.' + str(count) + ".jpg", cv2.resize(gray[y:y+h,x:x+w],(240,240)))
    
            return cv2.cvtColor(img,cv2.COLOR_BGR2RGB),no_user


