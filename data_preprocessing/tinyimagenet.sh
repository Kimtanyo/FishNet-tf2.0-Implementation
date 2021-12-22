#!/bin/bash

# download and unzip dataset
#wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip data/tiny-imagenet-200.zip -d data/

current="$(pwd)/data/tiny-imagenet-200"

cd $current
# remove txt files in current directory
# rm wnids.txt
# rm words.txt
#remove test directory
rm -r test

# training data
# remove detail label tx
# move all image files a level upward
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
# move val_annotation to the same level as val, train
mv val_annotations.txt ../
#put all images into val folder
mv images/* .
rm -r images
cd $current/val
# rename files
for f in *; do
    #padding digits in file name into 8 digits
    # add one to numbers so labelling start at 1 (not 0)
    mv "$f" "$(printf %08g.%s "$((${f//[^0-9]/}+1))" "${f##*.}")"
done
for f in *; do
    #append ILSVRC2012_val_ before existing file name
    mv "$f" ILSVRC2012_val_"$f"
done

# val_annotations.txt
cd $current
# only leave second columns 
cut -f 2 val_annotations.txt > output.txt
rm val_annotations.txt
mv output.txt val_annotations.txt

echo "done"
