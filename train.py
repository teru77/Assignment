import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-epoch',type=int, default=50,help='the number of epochs')
    parser.add_argument('-save_model', type=str,default='best_model.pth',help='the name of training model' )
    parser.add_argument('-id', default=0, type=int, help='the id of GPU(cuda)' )
    parser.add_argument('-save_folder', default="result", type=str, help='the folder name for saving results' )

    args = parser.parse_args()

    return args

def one_epoch(model, dataloader,device,criterion,optimizer= None):

        if optimizer:
            model.train()

        else:
            model.eval()

        loss_sum = 0.0
        acc_sum = 0.0
        data_num = 0
        iter_num = 0

        with tqdm(total=len(dataloader),unit='batch') as pbar:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                data_num += len(labels)
                iter_num += 1

                # training phase
                if optimizer:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # validation phase
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                loss_sum += loss.item()
                acc_sum += torch.sum(preds == labels.data).item()

                pbar.update(1)

        epoch_loss = loss_sum / iter_num
        epoch_acc = acc_sum / data_num

        return epoch_loss,epoch_acc

def result_plot(train_loss_list,valid_loss_list,train_accuracy_list,valid_accuracy_list,save_folder):
    fig= plt.figure(figsize=(10.0, 6.0))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(train_loss_list, 'b', label='train')
    ax1.plot(valid_loss_list, 'r', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(train_accuracy_list, 'b', label='train')
    ax2.plot(valid_accuracy_list, 'r', label='valid')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    ax2.legend()
    plt.savefig(f'./{save_folder}/result.png')

def train():

     # 1. Preprocessing
    args = parser()

    device = f'cuda:{args.id}' if (torch.cuda.is_available()) else 'cpu'
    print(device)

    num_epochs = args.epoch
    save_name = args.save_model
    save_folder = args.save_folder

    os.makedirs(save_folder,exist_ok=True)

    size = (224, 224)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 2. Preparation of datasets and dataloaders
    train_data_dir = './data/train'
    val_data_dir = './data/val'

    train_dataset =  torchvision.datasets.ImageFolder(train_data_dir, transform=data_transforms['train'])
    val_dataset=  torchvision.datasets.ImageFolder(val_data_dir, transform=data_transforms['val'])

    train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=5)
    class_names = train_dataset.classes

    # 3. Model preparation
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=len(class_names))

    # 4. Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 5. Training
    best_acc = 0.0
    train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list = [],[],[],[]
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch[{epoch+1}/{num_epochs}]')

        train_epoch_loss,train_epoch_acc = one_epoch(model,train_dataloader,device,criterion,optimizer)
        train_loss_list.append(train_epoch_loss)
        train_accuracy_list.append(train_epoch_acc)

        print(f'train loss: {train_epoch_loss}, train acc: {train_epoch_acc}')

        valid_epoch_loss,valid_epoch_acc = one_epoch(model,val_dataloader,device,criterion,)
        valid_loss_list.append(valid_epoch_loss)
        valid_accuracy_list.append(valid_epoch_acc)

        print(f'valid loss: {valid_epoch_loss}, valid acc: {valid_epoch_acc}')

        if valid_epoch_acc > best_acc:
            print('save model')
            torch.save(model.state_dict(), save_name)
            best_acc = valid_epoch_acc

    # 6. Plot the result
    result_plot(train_loss_list,valid_loss_list,train_accuracy_list,valid_accuracy_list,save_folder)

if __name__ == '__main__':
    train()