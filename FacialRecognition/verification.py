import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1    
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40,device="cuda:0") # keep_all=True to detect all faces in image.
resnet = InceptionResnetV1(pretrained='vggface2').eval() 

class Verification:
    def __init__(self,cam,load_data):
        self.load_data = load_data
        self.embedding_list = self.load_data[0] 
        self.name_list = self.load_data[1] 
        self.trust= 0
        self.cam = cam
        self.last_person = None
    def verification(self):        
        ret, frame = self.cam.read()
        frame = cv2.flip(frame, 1)
        img = np.array(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _ , points = mtcnn.detect(img,True)
            for i in points[0]:
                frame = cv2.circle(frame,(int(i[0]),int(i[1])),4,(0,0,255),-1)
            for i, prob in enumerate(prob_list):
                if prob > 0.97:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                    dist_list = []  # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(self.embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)
                    min_dist_idx = dist_list.index(
                            min_dist)
                        
                    name = self.name_list[min_dist_idx]

                    box = boxes[i]

                    if min_dist > 0.60:
                        frame = cv2.putText(frame, name+' '+str(round(min_dist, 3)), (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 0), 1, cv2.LINE_AA)
                        if self.last_person == name:
                            self.trust += 1
                        else:
                            self.trust = 0
                        self.last_person = name
                    
                    frame = cv2.rectangle(frame, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        return (cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),self.trust,self.last_person)
