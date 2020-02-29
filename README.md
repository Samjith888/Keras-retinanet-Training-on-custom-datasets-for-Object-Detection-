# Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-

## Prerequisites
Install keras-retinanet. It can be done in two ways:

    By using Fizyr keras-retinanet github repository. https://github.com/fizyr/keras-retinanet 
		                                    (or)
	pip install keras-retinanet 		
		
Note : Make sure that tensorflow is installed in the machine (tensorflow version < 2.0)
		
## Prepare Data for training	
1. Clone or download this repository.
2. Delete files inside snapshot,tensorboard,dataset/annotation and dataset/images in this directory. 
3. Paste all training images in dataset/images directory.
4. Next step is to create annotation file for each images and place all annotation xml files in dataset/annotations 
   directory. Which will be in xml format. LabelImg is one of the tool which can be used for annotation. 
	 [LabelImg github](https://github.com/tzutalin/labelImg) or [LabelImg exe](https://tzutalin.github.io/labelImg/)
5. Then have to set the config file custom_dataset_config.py inside config directory.
   Here set the path for annotation, image, train.csv files and also set the path where the classes.csv have to be 
	 saved. TRAIN_TEST_SPLIT value will split the data for training and testing.
6. create train.csv, test.csv and classes.csv by using build_dataset.py inside this directory.
    Run the following command inside the directory
		
		`python build_dataset.py`
		
    The datas inside train.csv and test.csv will be in the following format. 
		
		`<path/to/image>,<xmin>,<ymin>,<xmax>,<ymax>,<label>`
   
	  classes.csv will be in following format.
		
		 `label, index`
		 
The data is all set for training. Now lets start Training.
	
## Training	
Download the pretrained weights on the COCO datasets with resnet50 backbone from [this link](https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5). Paste it in the directory.

Start training by run the following one of the command. 

   `$ retinanet-train --weights resnet50_coco_best_v2.1.0.h5 --steps 400 --epochs 20 --snapshot-path snapshots 
	 --tensorboard-dir tensorboard csv dataset/train.csv dataset/classes.csv`
	 
or 
	`python keras_retinanet/bin/train.py --weights resnet50_coco_best_v2.1.0.h5 --steps 400 --epochs 20 
	--snapshot-path snapshots --tensorboard-dir tensorboard csv dataset/train.csv dataset/classes.csv`


![t1](https://user-images.githubusercontent.com/39676803/65702299-9e968b00-e037-11e9-9ce6-0809b6f329e8.JPG)	
	
After each epoch, the model will be saved in the snapshots folder. Stop the training by using 'ctr+c' when the loss will be lesser like in a range 0.1. 



## Convert Model

The model have to be converted in a format which can be used for prediction. For this run one of the following command.

   `$ retinanet-convert-model <path/to/desired/snapshot.h5> <path/to/output/model.h5>`
or

   `$ python keras_retinanet/bin/train.py <path/to/desired/snapshot.h5> <path/to/output/model.h5>`
	 
Note : If you set any anchor params (--config) during training, then you should pass it to the above command while model convertion. (eg : retinanet-convert-model <path/to/desired/snapshot.h5> <path/to/output/model.h5> --config config.ini).
Refer the 53 line (parser.add_argument('--config', help='Path to a configuration parameters .ini file.')) in keras-retinanet/bin/convert_model.py


## Start Detection

Set model_path, video_path, output_path and labels_to_names values in the people_detection_video.py 
Run the following command.


  `python people_detection_video.py`
	
 An output video will be saved in the mentioned path.
 
 Also detecting from image is done by run the following command.
 
   `python people_detection_image.py`
   
 Output of my object detection model. 
 
 ![detected_image](https://user-images.githubusercontent.com/39676803/65708379-4e252a80-e043-11e9-8791-61f7a6d135f3.jpg)


		 
