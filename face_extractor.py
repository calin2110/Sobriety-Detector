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
        self.read_df()

    def read_df(self):
        try:
            print("Reading dataset csv...")
            self.data_info = pd.read_csv(os.path.join(self.save_dir, 'data_info.csv'))
            print("Cleaning csv of deleted images...") 
            self.data_info = self.data_info[self.data_info['Name'].apply(
                lambda x: os.path.exists(os.path.join(self.save_dir, x))
                )]
            print("Data cleaned")
            self.image_index = 1+self.data_info['Name'].apply(lambda x: int(x.split('.')[0].split('_')[2])).max()
            print("Max index = " + str(self.image_index))
            print("Done") 
        except:
            print("Invalid csv, using new csv")
            self.data_info = pd.DataFrame()
    def save_df(self):
        print("Saving data info")
        self.data_info.to_csv(os.path.join(self.save_dir, 'data_info.csv'), index=False)

    def extract_from(self, video_path, label = 0):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(video_path)
 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        frame_nr = 0

        label_name = "sober_"
        if label:
            label_name = "drunk_"

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
                print("Frame nr: {}/{}".format(frame_nr, length), end='\r')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_name = "face_" + label_name + str(self.image_index) + ".jpg"
                img_path = os.path.join(self.save_dir, img_name) 
                im = Image.fromarray(frame)
                im.save(img_path)
                try:
                    face = DeepFace.detectFace(img_path = img_path, target_size = (224, 224), detector_backend = "dlib")

                    im = Image.fromarray((face*255).astype(np.uint8))
                    im.save(img_path)
                    #print("New Face Detected: " + str(self.image_index))
                    self.data_info = self.data_info.append({
                        'Name': img_name,
                        'Label': label
                        }, ignore_index=True)
                    self.image_index += 1
                except ValueError as e:
                    #print("No Face Detected")
                    os.remove(img_path)
                
                frame_nr += self.frame_cut
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
            else:
                break
        cap.release()
    def extract_from_tiktok(self, video_path, start_label = 0):
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(video_path)
 
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        frame_nr = 0
        label = start_label
        last_frame = None

        epsilon = 0.2

        label_name = "sober_"

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (last_frame is not None) and abs(1 - norm(frame) / norm(last_frame)) > epsilon:
                    label = 1 - label
                    if not label:
                        label_name = "sober_"
                        #print(self.image_index, "Sober")
                    else:
                        label_name = "drunk_"
                        #print(self.image_index, "Drunk")
                frame_nr += 1
                print("Frame nr: {}/{}".format(frame_nr, length), end='\r')
                last_frame = frame.copy()
                if frame_nr % self.frame_cut != 0:
                    continue
                img_name = "face_" + label_name + str(self.image_index) + ".jpg"
                img_path = os.path.join(self.save_dir, img_name) 
                im = Image.fromarray(frame)
                im.save(img_path)
                try:
                    face = DeepFace.detectFace(img_path = img_path, target_size = (224, 224), detector_backend = "dlib")

                    im = Image.fromarray((face*255).astype(np.uint8))
                    im.save(img_path)
                    #print("New Face Detected: " + str(self.image_index))
                    self.data_info = self.data_info.append({
                        'Name': img_name,
                        'Label': label
                        }, ignore_index=True)
                    self.image_index += 1
                except ValueError as e:
                    #print("No Face Detected")
                    os.remove(img_path)
            else:
                break
        cap.release()

extractor = FaceExtractor(save_dir="data_test", frame_cut=5)
drunk_dir = "videos/Drunk"
sober_dir = "videos/Sober/"
drunk_videos = os.listdir(drunk_dir)
#for i, filename in enumerate(drunk_videos):
#    f = os.path.join(drunk_dir, filename)
#    # checking if it is a file
#    if os.path.isfile(f):
#        print("Starting extraction from {} || {}/{}".format(f, i, len(drunk_videos)))
#        extractor.extract_from(f, label=1)
#        print("Done                                 ")
#        extractor.save_df()
#print("Drunk Videos done")

sober_videos = os.listdir(sober_dir)
for i, filename in enumerate(sober_videos):
    f = os.path.join(sober_dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print("Starting extraction from {} || {}/{}".format(f, i, len(sober_videos)))
        extractor.extract_from(f, label=0)
        print("Done                                 ")
        extractor.save_df()
print("Sober Videos done")
