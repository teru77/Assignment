import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', type=str,default='best_model.pth',help='the name of training model' )
    parser.add_argument('-id', default=0, type=int, help='the id of GPU(cuda)' )
    parser.add_argument('-save_folder', default="result", type=str, help='the folder name for saving results' )

    args = parser.parse_args()

    return args


def test():

     # 1. Preprocessing
    args = parser()
    device = f'cuda:{args.id}' if (torch.cuda.is_available()) else 'cpu'
    print(device)

    save_folder = args.save_folder
    os.makedirs(save_folder,exist_ok=True)

    size = (224, 224)
    transform =  transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # 2. Preparation of a dataset and a dataloader
    test_data_dir = './data/test'
    test_dataset =  torchvision.datasets.ImageFolder(test_data_dir, transform=transform)

    dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    class_names = test_dataset.classes

    # 3. Model preparation
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=len(class_names))
    model.load_state_dict(torch.load(args.model))
    model = model.to(device)

    # 4. Test
    acc_sum = 0.0
    data_num = 0
    pred_list = []
    label_list = []

    with tqdm(total=len(dataloader),unit='batch') as pbar:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            data_num += len(labels)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            acc_sum += torch.sum(preds == labels.data).item()
            pred_list += preds.detach().cpu().numpy().tolist()
            label_list += labels.detach().cpu().numpy().tolist()

            pbar.update(1)

    acc = acc_sum / data_num
    print(f'test_accuracy: {acc}')

    # 5. Create confusion matrix
    cm = confusion_matrix(label_list,pred_list)
    cm = pd.DataFrame(data=cm, index=['good', 'not-good'],
                           columns=['good', 'not-good'])
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.yticks(rotation=0)
    plt.xlabel('Predection', fontsize=10, rotation=0)
    plt.ylabel('Ground truth', fontsize=10)
    plt.savefig(f'./{save_folder}/confusion_matrix.png')


if __name__ == '__main__':
    test()