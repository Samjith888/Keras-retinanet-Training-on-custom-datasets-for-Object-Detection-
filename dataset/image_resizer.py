
from PIL import Image
import os

path = "C:\\Users\\Samjith.CP\\Desktop\\terrorist_detect\\dataset\\images\\"
output= "C:\\Users\\Samjith.CP\\Desktop\\terrorist_detect\\dataset\\"


def resize():
    dirs = os.listdir(path)
    for img in dirs:
        image = Image.open(path+img)
        name, ext = os.path.splitext(img)
        img_resize = image.resize((650, 420), Image.ANTIALIAS)
        img_resize.save(output+name+ext, quality=100)


resize()