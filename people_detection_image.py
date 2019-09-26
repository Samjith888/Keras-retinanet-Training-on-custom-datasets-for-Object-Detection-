
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())

model_path = 'C:\\Users\\Samjith.CP\\Desktop\\test.h5'    ## replace this with your model path
model = models.load_model(model_path, backbone_name='resnet50')
labels_to_names = {0: 'person'}                    ## replace with your model labels and its index value

image_path = 'C:\\Users\\Samjith.CP\\Desktop\\first_terrorist_detect\\dataset\\images\\4.jpg'  ## replace with input image path
output_path = 'C:\\Users\\Samjith.CP\\Desktop\\detected_image.jpg'   ## replace with output image path

def detection_on_image(image_path):

        image = cv2.imread(image_path)

        draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            if score < 0.4:
                break

            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, detected_img)
        cv2.imshow('Detection',detected_img)
        cv2.waitKey(0)
detection_on_image(image_path)