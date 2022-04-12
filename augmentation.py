import argparse
import os
from PIL import Image

def augmentation():
    parser = argparse.ArgumentParser()

    parser.add_argument('path',type=str, help='the path of the image folder' )
    args = parser.parse_args()

    path = args.path
    img_dir = os.listdir(path)

    for file in img_dir:
        img = Image.open(os.path.join(path,file))
        img_1 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_2 = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_3 = img.transpose(Image.ROTATE_180)

        img_1.save(f'{path}/1_{file}')
        img_2.save(f'{path}/2_{file}')
        img_3.save(f'{path}/3_{file}')

    print("Augmentation Complete!!")

if __name__ == '__main__':
    augmentation()