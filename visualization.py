import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class Visualization:
    def __init__(self, m):
        self.model = m
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
    
    def grad_cam(self, imgs_path: list):
        gradcam = GradCAM(model=self.model,target_layers=[self.model.model.layer4[-1]])
        self.model.eval()
        grad_cam_maps = []

        for img_path in imgs_path:
            transform = T.Compose([T.ToTensor()])
            img = Image.open(img_path)
            np_img = np.asarray(img) / 255

            X = transform(img)
            X = X.unsqueeze(0)
            X.requires_grad_()

            grayscale_cam = gradcam(input_tensor=X)
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
            grad_cam_maps.append((img, visualization))
        return grad_cam_maps


    def saliency_map(self, imgs_path: list):
        self.model.eval()
        saliency_maps = []

        for img_path in imgs_path:
            transform = T.Compose([T.ToTensor()])
            img = Image.open(img_path)

            X = transform(img)
            X = X.unsqueeze(0)

            X.requires_grad_()

            output = self.model(X)

            max_index = output.argmax()
            output = output[0,max_index]

            output.backward()

            saliency, _ = torch.max(X.grad.data.abs(),dim=1)

            saliency_maps.append((img, saliency[0]))
        
        return saliency_maps
    
    @staticmethod
    def show_img_and_vis(imgs_and_vis, visualization_label: str):
        fig = plt.figure(figsize=(7,5))
        rows = len(imgs_and_vis)
        columns = 2

        for i in range(rows):
            img, saliency = imgs_and_vis[i]
            fig.add_subplot(rows, columns, (i+1) * 2 - 1)
            plt.imshow(img, cmap=plt.cm.hot)
            plt.axis('off')
            plt.title("Original")

            fig.add_subplot(rows, columns, (i+1) * 2)
            plt.imshow(saliency, cmap=plt.cm.hot)
            plt.title(visualization_label)
            plt.axis('off')

        plt.show()