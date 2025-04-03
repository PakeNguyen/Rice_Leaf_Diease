import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize  
from torch.utils.data import DataLoader
import argparse
import shutil
import matplotlib.pyplot as plt

from torchvision.models import resnet34 ,resnet50 , ResNet34_Weights, ResNet50_Weights

from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import cv2
import warnings
warnings.filterwarnings("ignore")

def get_args():
        parse = argparse.ArgumentParser()
        parse.add_argument("--image_size", "-i", type=int, default=224)
        parse.add_argument("--image_path", "-p", type=str, default="C:/Hoc_May/All_Project/data_lua/leaf_blast_1.jpg")
        parse.add_argument("--checkpoint_path", "-c", type=str, default="C:/Hoc_May/All_Project/checkpoint/lua")
        args = parse.parse_args()

        return args

def inference(args):
    classes = ["bacterial_leaf_blight", "healthy", "leaf_blast", "leaf_scald", "narrow_brown_spot", 
                           "rice_hispa", "sheath_blight"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (args.image_size, args.image_size))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

      # tương đương ToTensor() từ Pytorch
    image = image / 255.
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)

    model = resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=7)

    checkpoint = torch.load(os.path.join(args.checkpoint_path, "best.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()
    with torch.no_grad():
      output = model(image)
      softmax = nn.Softmax()
      prog = softmax(output)
      predicted_prob, predicted_classes = torch.max(prog, dim=1)

      score = predicted_prob*100
      img_show = cv2.imread(args.image_path)
      img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

      plt.imshow(img_show)
      plt.title(f"({classes[predicted_classes]}) - Accuracy: {score[0]:.2f}%")
      plt.axis('off')
      plt.show()
      
if __name__ == "__main__":
    args = get_args()
    inference(args)
    