Translations: [简体中文](README.md) | [English](README-en.md) 

# Introduce
FEG(Facial Expression Generation) automatically takes screenshots of the faces in the photos, generates gif facial expression animations. generates expression  based on specific expressions video templates. Using this project, you can use any face photo to generate a variety of expressions.

Based on [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model).


## Example
![avatar-imitator](data/example/example1.gif)
![avatar-imitator](data/example/example2.gif)
![avatar-imitator](data/example/example3.gif)



## Quick Start

0.recommand enviornment：python3.6.9 on ubuntu 18.04


1.ubuntu python3+ ,execute following instructions:
```
apt update
apt install -y libgl1-mesa-glx ffmpeg gifsicle
pip3 install --upgrade pip
git clone https://github.com/techtogood/feg
cd feg
pip3 install -r requirements.txt
```


2.download the checkpoint to dir: model/avatarify_models
[baidu-disk](https://pan.baidu.com/s/1O7K-s0oaevmF8zmLayU74Q) code:z846 
[google-drive](https://drive.google.com/file/d/1rMz7HO-znqLaW1hm_hBHQwhrAgmC6Krg/view?usp=sharing)

3.run:
```
python3 app.py --source_image data/input/Monalisa.png  --driving_video data/imitator_video/smile.mp4  --text nice
```
parameters：
source_image：source face image
driving_video：expression video model
text:text on the gif animation


## YAML

Project config ，defualt run on CPU. If support GPU，run on GPU：
```config/config.yaml```

avatarify model config
```vox-256.yaml```



