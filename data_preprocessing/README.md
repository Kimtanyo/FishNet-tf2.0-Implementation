# How to Preprocess Tiny ImageNet Data


## Preprocess Structures
Since the dataset is large, you should keep them in local instead of pushing it to the repo.

- First, download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip, save it to `data_preprocessing/data/tiny-imagenet-200.zip`. 
- Then you could run the bash file `tinyimagenet.sh`. You could simply click and run or in terminal `./tinyimagenet.sh`. 
- The preprocessed file are saved in `data_preprocessing/data/tiny-imagenet-200`, the tree structure looks like 

```
data_preprocessing
├── README.md
├── data
│   ├── tiny-imagenet-200
│   │   ├── train
│   │   │   ├── n01443537
│   │   │   │   ├── n01443537_0.JPEG
│   │   │   │   ├── ...
│   │   │   │   └── n01443537_99.JPEG
│   │   │   ├── ...
│   │   │   └── n12267677
│   │   │       └── ...
│   │   ├── val
│   │   │   ├── ILSVRC2012_val_00000001.JPEG
│   │   │   ├── ...
│   │   │   └── ILSVRC2012_val_00010000.JPEG
│   │   └── val_annotations.txt
│   └── tiny-imagenet-200.zip
└── tinyimagenet.sh

204 directories, 110004 files
```

## Convert to TFRecord Dataset

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