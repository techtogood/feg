Translations: [简体中文](README.md) | [English](README-en.md) 

# 介绍
FEG(Facial Expression Generation)项目自动动截图照片中人脸，生成gif人脸表情动画，生成的表情基于选定的表情视频模板。基于本项目，可以使用任意人脸照片生成各种各样有趣的表情，
 欢迎Start！  
   
 人脸活化模型基于： [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model).

## 示例
![FEG](data/example/example1.gif)
![FEG](data/example/example2.gif)
![FEG](data/example/example3.gif)

## 更多请访问
### 网站：
[ffmagic.com](https://www.ffmagic.com)

### 微信小程序：
![FEG](data/img/wetchat_miniapp.jpg)


## 快速开始

推荐环境：python3.6.9 on ubuntu 18.04


ubuntu系统，python3以上环境，执行以下命令安装项目系统依赖:
```
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx ffmpeg gifsicle
pip3 install --upgrade pip
git clone https://github.com/techtogood/feg
cd feg
pip3 install -r requirements.txt
```


下载预训练模型到 model/avatarify_models 目录:

[baidu-disk](https://pan.baidu.com/s/1O7K-s0oaevmF8zmLayU74Q) 提取码:z846   
[google-drive](https://drive.google.com/file/d/1rMz7HO-znqLaW1hm_hBHQwhrAgmC6Krg/view?usp=sharing)


运行：
```
python3 app.py --source_image data/input/Monalisa.png  --driving_video data/imitator_video/smile.mp4  --text nice
```
source_image：源人脸图片
driving_video：模仿的人脸表情视频
text:gif表情动画的文字


## YAML 配置

项目工程配置，算法默认运行在CPU，如运行环境支持GPU，则算法跑到GPU：
```config/config.yaml```

avatarify模型配置:
```vox-256.yaml```


### 联系我们（微信公众号）：

![FEG](data/img/wechat_official_account.jpg)

