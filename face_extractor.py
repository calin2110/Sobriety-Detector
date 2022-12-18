import cv2
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace
import os
from PIL import Image
import pandas as pd

class FaceExtractor():
    def __init__(self, save_dir = "data", frame_cut = 60):
        self.save_dir = save_dir
        self.image_index = 0
        self.frame_cut = frame_cut
        self.person_index = 0
        # Dataframe to be created, video path and label 0 for sober 1 for drunk
        self.data_info = pd.DataFrame()
    def extract_from(self, video_path, start_label = 0):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(video_path)
 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        frame_nr = 0
        label = start_label
        last_frame = None

        epsilon = 0.2

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (last_frame is not None) and abs(1 - norm(frame) / norm(last_frame)) > epsilon:
                    label = 1 - label
                    if not label:
                        print(self.image_index, "Sober")
                    else:
                        print(self.image_index, "Drunk")
                frame_nr += 1
                last_frame = frame.copy()
                if frame_nr % self.frame_cut != 0:
                    continue
                img_path = self.save_dir + "/face_" + str(self.image_index) + ".jpg" 
                im = Image.fromarray(frame)
                im.save(img_path)
                try:
                    face = DeepFace.detectFace(img_path = img_path, target_size = (224, 224), detector_backend = "dlib")

                    im = Image.fromarray((face*255).astype(np.uint8))
                    im.save(img_path)
                    print("New Face Detected: " + str(self.image_index))
                    self.data_info = self.data_info.append({
                        'Video Name': "face_" + str(self.image_index) + ".jpg",
                        'Label': label
                        }, ignore_index=True)
                    self.image_index += 1
                except ValueError as e:
                    print("No Face Detected")
                    os.remove(img_path)
            else:
                break
        cap.release()

extractor = FaceExtractor(frame_cut=5)
extractor.extract_from("video_1.mp4")
extractor.data_info.to_csv('data_info.csv')
