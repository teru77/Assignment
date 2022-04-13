# Assignment
This is the repository for intern assignments.
* Dataset : MVTec Screws dataset
* Framework : Pytorch (Python)
* ML model : VGG-16

## Development environment
* OS : Linux (Ubuntu 18.04.5 LTS)
* GPU : Quadro RTX 6000
* VRAM : 24GB
* RAM : 32GB
* Execution environment : Anaconda

## Set up
```shell
conda create -n assignment python=3.10
conda activate assignment

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
(If you are using an OS other than Linux, please select the appropriate version for your OS.Please note that you may need to specify paths and other settings.)

git clone https://github.com/teru77/Assignment.git
cd Assignment
pip install -r requirements.txt

mkdir data
```

## Directory structure
```none
├── .gitignore
├── data
│   ├── train
│   │   ├── good
│   │   ├── not-good
│   ├── val
│   │   ├── good
│   │   ├── not-good
│   ├── test
│   │   ├── good
│   │   ├── not-good
├── train.py
├── test.py
├── augmentation.py
├── requirements.txt
```
Please put the images in the data folder so that they fit the above structure. <br />
Train and val are divided 8:2 for train images.

## Usage
### train
```shell
python train.py -epoch 50 -save_model best_model.pth -id 0 -save_folder result
```
### test
```shell
python test.py -model best_model.pth -id 0 -save_folder result
```

## Augmentaion
The following three types of images are created.
![図1](https://user-images.githubusercontent.com/64674323/162917982-99d21894-5468-4983-a57f-e7df0d4f0f83.png)

To use data augmentation, please run the following command.
```shell
python augmentation.py {PATH}
```
PATH should be the folder where the data augmentation will take place. The images are saved in that folder.
Each file name is prefixed with the number in () shown in the figure.
