import cv2
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("FacialRecognition/haarcascade_frontalface_default.xml")
import torch.optim as optim
def train_model(model,data_train,data_val):
  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')



# function to get the images and label data
class Data(Dataset):
    def __init__(self, path):
        self.path = path
        self.imagePaths = [os.path.join(self.path,f) for f in os.listdir(self.path)]  
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):   
        PIL_img = Image.open(self.imagePaths[idx]).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(self.imagePaths[idx])[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            img= img_numpy[y:y+h,x:x+w]/255.0
        return img,id
    
    
data_train = Data("dataset/train/pics")
data_val = Data("dataset/val/pics")
data_test = Data("dataset/test/pics")

train_dataloader = DataLoader(data_train, batch_size=64, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=64, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)
from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x= F.sigmoid(x)
        return x


model = Net()

train_model(model,train_dataloader,val_dataloader)

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")










#faces,ids = getImagesAndLabels(path)
#recognizer.train(faces, np.array(ids))


#recognizer.write('trainer.yml') # recognizer.save() worked on Mac, but not on Pi


#print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
