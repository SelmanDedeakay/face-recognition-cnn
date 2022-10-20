from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

def main():
    mtcnn = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=150,device="cuda")#keep_all=False to detect single face in image.
    resnet = InceptionResnetV1(pretrained='vggface2').eval() 
    dataset = datasets.ImageFolder('photos') # photos folder path 
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

    def collate_fn(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_fn)

    name_list = [] # list of names correspoing to cropped photos
    embedding_list = [] # list of embedding matrix after conversion from cropped faces to embedding matrix using resnet

    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True) 
        if face is not None and prob>0.97:
            emb = resnet(face.unsqueeze(0)) 
            embedding_list.append(emb.detach()) 
            name_list.append(idx_to_class[idx])        


    data = [embedding_list, name_list] 
    torch.save(data, 'data.pt')