# FishNet tf2.0 Implementation

This repo holds the Tensorflow 2.x implementation of the paper:

[FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf) , Shuyang Sun, Jiangmiao Pang, Jianping Shi, Shuai Yi, Wanli Ouyang, NeurIPS 2018.

## Prerequisites

- Python 3.8
- Tensorflow 2.7
- cuda 11.2
- cudnn 8.1

## Dataset

Our implementation is based on [Tiny ImageNet](http://cs231n.stanford.edu/reports/2016/pdfs/401_Report.pdf). The dataset contains 100,000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, 50 validation images, and 50 test images.

We've already stored the original Tiny ImageNet dataset and the TFRecord files after preprocessing in google drive [here](https://drive.google.com/drive/folders/1PQOq3dVTjHKsao5lD0RvbdUQarAGstzk). Thus, if you want to train FishNet on the same dataset, you can skip the preprocessing below and use our data directly.

### Preprocess Structures

Since the dataset is large, you should keep them in local instead of pushing it to the repo.

- First, download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip, save it to `data_preprocessing/data/tiny-imagenet-200.zip`. 
- Then you could run the bash file `tinyimagenet.sh`. You could simply click and run or in terminal `./tinyimagenet.sh`. 
- The preprocessed file are saved in `data_preprocessing/data/tiny-imagenet-200`, the tree structure looks like 

```
data_preprocessing
├── README.md
├── data
│   ├── tiny-imagenet-200
│   │   ├── train
│   │   │   ├── n01443537
│   │   │   │   ├── n01443537_0.JPEG
│   │   │   │   ├── ...
│   │   │   │   └── n01443537_99.JPEG
│   │   │   ├── ...
│   │   │   └── n12267677
│   │   │       └── ...
│   │   ├── val
│   │   │   ├── ILSVRC2012_val_00000001.JPEG
│   │   │   ├── ...
│   │   │   └── ILSVRC2012_val_00010000.JPEG
│   │   └── val_annotations.txt
│   └── tiny-imagenet-200.zip
└── tinyimagenet.sh

204 directories, 110004 files
```

### Convert to TFRecord Dataset

Please refer to [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets) and [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) for following process on converting ImageNet dataset to TFRecord format. We have a copy of script you can use in this repo.

- If you follow the steps above, you now should have the preprocessed raw data in the above structure. 
- You could choose to store TFRecords on Google Cloud Storage or locally. For our purpose, we store datasets to `data_preprocessing/data/tf_records` by running the following terminal code. 
```
python3 imagenet_to_gcs.py \
  --raw_data_dir=data/tiny-imagenet-200 
  --local_scratch_dir=data/tf_records 
  --nogcs_upload
```
NOTE: Even if you don't need to upload to GCS, make sure you have `pip install gcloud google-cloud-storage` to avoid compiling errors.
- If your folder structures are different, please change the following line to fit your structure.
```
LABELS_FILE = [YOUR VALIDATION ANNOTATION]

TRAINING_SHARDS = 128
VALIDATION_SHARDS = 64

TRAINING_DIRECTORY = [YOUR TRAINING SET PATH]
VALIDATION_DIRECTORY = [YOUR VALIDATION SET PATH]
```

## Training
Note our fishnet code orginates from paper's authors pytorch fishnet version
https://github.com/kevin-ssy/FishNet

1. You should open `main.ipynb` and run all commands including importing, initializing global variables and defining funtions of preprocessing, creating models and training. 

2. Then you can train ImageNet dataset by the following command (also included in the ipynb file)

   ```python
   model = create_model()
   train(model, 0)
   # 1 loop = 5 epochs
   maxloop=6
   for i in range(1,maxloop):
       #check out saved checkpoints
       checkpoint_path = "training_"+str(i-1)+"/cp.ckpt"
       checkpoint_dir = os.path.dirname(checkpoint_path)
       os.listdir(checkpoint_dir)
       #create a model
       model = create_model()
       #load trained weights
       model.load_weights(checkpoint_path)
   	#training next loop
       model = train(model, i)
   ```

   Note that if you execute accurate grid search on hyper-parameters which controls the number of trainable parameters in FishNet, you can get a much better resutlt than ours.

## Key Functions Explanation

`main.ipynb`

| Function                     | Explanation                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| parse_record(record)         | output parsed tfrecord based on features                     |
| preprocess_data(is_training) | output training/val X,y tuple ready to feed into models      |
| train(model, trained_times)  | output model trained by 5 more epochs and save the  checkpoints |

Folder `model_tf2`

| .py file       | Class/Function      | Explanation                                          |
| -------------- | ------------------- | ---------------------------------------------------- |
| fish_block.py  | Class Bottleneck    | Bottleneck Residual Unit                             |
| fishnet.py     | Class Fish          | main FishNet structure including tail, body and head |
| net_factory.py | myfishnet(**kwargs) | FishNet with medium order of magnitude of parameters |

```
## Organization of this directory./
├── README.md
├── data_preprocessing
│   ├── README.md
│   ├── ReadingRecord.ipynb
│   ├── imagenet_to_gcs.py
│   └── tinyimagenet.sh
├── figure
│   ├── hard_model_accuracy.png
│   ├── hard_model_loss.png
│   ├── medium_model_accuracy.png
│   └── medium_model_loss.png
├── main.ipynb
└── model_tf2
    ├── __init__.py
    ├── fish_block.py
    ├── fishnet.py
    └── net_factory.py

3 directories, 14 files
```
