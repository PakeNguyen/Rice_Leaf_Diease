import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
import argparse
import shutil
import matplotlib.pyplot as plt
from dataset_Lua import Lua_dataset
from torchvision.models import resnet34, resnet50 , ResNet34_Weights, ResNet50_Weights
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def get_args():
        parse = argparse.ArgumentParser()
        parse.add_argument("--data_path", "-d", type=str, default="C:/Hoc_May/All_Project/data_lua/Rice_Leaf_Diease")
        parse.add_argument("--epochs", "-e", type=int, default=100)
        parse.add_argument("--batch_size", "-b", type=int, default=32)
        parse.add_argument("--image_size", "-i", type=int, default=224)
        parse.add_argument("--lr", "-l", type=float, default=1e-2)    
        parse.add_argument("--log_path", "-p", type=str, default="C:/Hoc_May/All_Project/tensorboard/lua")
        parse.add_argument("--checkpoint_path", "-c", type=str, default="C:/Hoc_May/All_Project/checkpoint/lua")
        args = parse.parse_args()

        return args


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=2048, out_features=7)
    model.to(device)

    transforms = Compose([
        ToTensor(),
        Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        Resize((args.image_size, args.image_size))
    ])
    train_dataset = Lua_dataset(root=args.data_path, is_train=True, transforms = transforms)
    Train_DataLoader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False
    )

    test_dataset = Lua_dataset(root=args.data_path, is_train=False, transforms=transforms)
    Test_DataLoader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        drop_last=False
    )
    
    criterion = nn.CrossEntropyLoss()
    otimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if os.path.isdir(args.log_path):
         shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.checkpoint_path):
         os.makedirs(args.checkpoint_path)
    
    write = SummaryWriter(args.log_path)
    best_accuracy = 0

    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        progress_Bar = tqdm(Train_DataLoader, colour="cyan")
        for i, (images, labels) in enumerate(progress_Bar):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)
            progress_Bar.set_description(f"Epochs {epoch}/{args.epochs}. Loss {loss:0.4f}")
            write.add_scalar("Train/Loss", loss, epoch*(len(Train_DataLoader)) + i)
            otimizer.zero_grad()
            loss.backward()
            otimizer.step()

        # VALIDATION
                # VALIDATION
        model.eval()
        all_losses = []
        all_labels = []
        all_predictions = []

        import time
        from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
        from sklearn.preprocessing import label_binarize

        start_time = time.time()  # bắt đầu đo inference time

        with torch.no_grad():  
            for images, labels in Test_DataLoader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = criterion(output, labels)
                all_losses.append(loss.item())
                predictions = torch.argmax(output, dim=1)
                
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())

        end_time = time.time()  # kết thúc đo inference time

        # === METRICS ===
        Accuracy = accuracy_score(all_labels, all_predictions)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        loss_mean = np.mean(all_losses)

        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        # Tính mAP (Classification mAP)
        all_labels_bin = label_binarize(all_labels, classes=list(range(10)))
        all_predictions_bin = label_binarize(all_predictions, classes=list(range(10)))
        map_score = average_precision_score(all_labels_bin, all_predictions_bin, average='macro')

        # Inference time
        total_images = len(Test_DataLoader.dataset)
        inference_time = (end_time - start_time) / total_images
        fps = 1.0 / inference_time

        # === Logging ===
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Loss: {loss_mean:.4f} | Accuracy: {Accuracy:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | mAP: {map_score:.4f}")
        print(f"Inference time per image: {inference_time:.4f}s | FPS: {fps:.2f}")

        write.add_scalar("Test/Loss", loss_mean, epoch)
        write.add_scalar("Test/Accuracy", Accuracy, epoch)
        write.add_scalar("Test/Precision", precision, epoch)
        write.add_scalar("Test/Recall", recall, epoch)
        write.add_scalar("Test/F1_Score", f1, epoch)
        write.add_scalar("Test/mAP", map_score, epoch)
        write.add_scalar("Test/Inference_time", inference_time, epoch)
        write.add_scalar("Test/FPS", fps, epoch)

        plot_confusion_matrix(write, conf_matrix, train_dataset.categories, epoch)

        # === Save checkpoint ===
        checkpoint = {
            "model_state_dict" : model.state_dict(),
            "epoch" : epoch,
            "optimizer_state_dict" : otimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if Accuracy > best_accuracy:
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
            best_accuracy = Accuracy

if __name__ == "__main__":
    args = get_args()
    train(args)
    