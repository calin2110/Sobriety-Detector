# Sobriety Detector from Images

This project is aimed at detecting drunkenness and sobriety from images using a deep learning approach. The technique used is transfer learning with the ResNet 50 pretrained model. Our method involves collecting a dataset using viral videos of the ”first drink and last drink” trend, and extracting faces from several frames. The dataset is then used to fine-tune the ResNet 50 model by replacing the last fully connected layer with three fully connected layers.

The __SobrietyModel__ class was written with the help of ChatGPT for modifying the pre-trained model for transfer learning

The __SobrietyDataset__ and __SobrietyTrainer__ classes were written based on the Pytorch documentation (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

The __FaceExtractor__ class and the __data_split__ script were written by us, since there is no other dataset generation method available for our needs.

The __Visualization__ class was written half by us (Saliency Maps) and half with the help of the documentation of the 3rd party GradCAM library (https://pypi.org/project/grad-cam/).
