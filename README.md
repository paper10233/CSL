# CSL
We will add complete codes of CSL in the future.

# Installation
Install dependencies. We use python 3.7 and pytorch >= 1.7.0

```conda create -n CSL
conda activate CSL
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${CSL_ROOT}
pip install cython
pip install -r requirements.txt
```

We use DCNv2 in our backbone.
```
cd DCNv2
./make.sh
```

# Data Preparation

We use MOT15, MOT17 and MOT20 to train and evaluate our model. The corresponding datasets can be downloaded from the official webpage of MOTchallenge.
After downloading, you should prepare the data in the following structure:

```
MOT15
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```
Then, you can change the seq_root and label_root in src/gen_labels_X.py and run it, where x is dataset version.

# Training and Tracking
pretrained models: comming soon.

train on MOT15
```sh experiments/mot15_dla34.sh```

train on MOT17
```sh experiments/mot17_dla34.sh```

train on MOT20
```sh experiments/mot20_dla34.sh```


