from concurrent.futures import thread
import PIL.ImageTk
import PIL.Image
import tkinter as tk
import cv2
import cv2
import verification
import dataset
from tkinter import simpledialog,messagebox
import training 
import torch
import threading

cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.vid_dataset = dataset.Dataset(cam)
        self.canvas = tk.Canvas(
            window, width=600, height=480)
        self.canvas.pack()
        self.container = self.canvas.create_image(0, 0, anchor="nw")
        self.flag = None
        self.button_verificate = tk.Button(
            self.window, text="Verification", command=self.verification,width=100)
        self.button_verificate.pack()
        self.button_new_user = tk.Button(
            self.window, text="New User", command=self.new_user,width=100)
        self.button_new_user.pack()
        self.no_user = False

        self.button_train = tk.Button(
            self.window, text="Train Model", command=self.train,width=100)
        self.button_train.pack()
        self.button_train["state"] = "disabled"
        try:
            load_data = torch.load("C:\\Users\\Selman\\Desktop\\DL\\face-recognition-cnn\\FacialRecognition\\data.pt")
        except:
            self.no_user = True
            messagebox.showerror("Error", "No recorded face found. Please use New User button.")
            self.button_verificate["state"] = "disabled"
        self.delay = 10
        self.count = 0
        self.face_id = "unknown"
        self.window.mainloop()
    
    def verification(self):
        load_data = torch.load("C:\\Users\\Selman\\Desktop\\DL\\face-recognition-cnn\\FacialRecognition\\data.pt")
        self.vid_verification = verification.Verification(cam,load_data)     
        self.flag = False
        self.button_verificate["state"] = "disabled"
        self.button_new_user["state"] = "normal"
        self.predict()

    def predict(self):
        (frame, trust, last_person) = self.vid_verification.verification()
        if trust == 20:
            print(f"Bu kişi {last_person}")
        self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(frame))
        self.canvas.itemconfigure(self.container, image=self.photo)
        if self.flag==False:
            self.window.after(self.delay, self.predict)

    def new_user(self):
        self.face_id = simpledialog.askstring(
            "New User", "Enter a name for the new user:")
        
        if self.face_id is not None:
            self.flag = True
            self.count=0
            self.button_new_user["state"] = "disabled"
            if not self.no_user:
                self.button_verificate["state"] = "normal"
            
            self.dataset()

    def dataset(self):
        try:
            frame,no_user = self.vid_dataset.dataset(self.face_id,self.no_user,self.count)
            self.no_user = no_user
            self.photo = PIL.ImageTk.PhotoImage(
            image=PIL.Image.fromarray(frame))
            self.canvas.itemconfigure(self.container, image=self.photo)
        except Exception as e:
            pass
        finally:
            if self.count==self.vid_dataset.limit:
                self.button_train["state"] = "normal"
            if self.flag==True:
                self.count+=1
                self.window.after(self.delay, self.dataset)
                
    def train(self):
        App.TrainThread([self.button_new_user,self.button_verificate,self.button_train],self.flag).start()

    class TrainThread(threading.Thread):
        def __init__(self,button,flag):
            threading.Thread.__init__(self)
            self.buttons = button
            self.flag = flag
        def run(self):
            self.buttons[2]["text"] = "Training..."
            for button in self.buttons:
                button["state"] ="disabled"
            training.main()
            if self.flag==True:
                self.buttons[1]["state"] ="normal"
            else:
                self.buttons[0]["state"] ="normal"
            self.buttons[2]["text"] = "Train Model"

            
App(tk.Tk(), "App")
