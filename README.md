# DeepLearning
DeepLearning Basic
딥러닝의 기초인 뉴런의 Perception 부터 이미지 데이터를 지나 자연어 처리까지 많이 알려진 모델들을 만들고 학습시킵니다. 

최종적으로는 마스크를 착용했는지 안했는지 분류해주는 모델을 만듭니다.

## Overview

### 1.Perceptron

tensorflow 를 이용하여 input, weight, hypothesis 를 계산해보고 적절한 활성화 함수(relu, sigmoid)를 사용해봅니다.

![image](https://user-images.githubusercontent.com/40593455/98448044-b910c280-216c-11eb-97e6-6fe76405d41b.png)


### 2_MNIST

0~9 까지의 숫자를 구별 해주는 classifier model을 생성합니다. MNIST 데이터셋을 가지고 MLP를 적용하며 Layer 수를 조정하고 가중치를 초기화 하는 방식을 사용해보고 (Xaiver)
오버피팅을 방지하는 테크닉을 연구한다.(Dropout, Weight decay, Weight normalization)

![image](https://user-images.githubusercontent.com/40593455/98448092-1c025980-216d-11eb-8f71-ced24a831761.png)

### 3_LeNet_CIFA10

Convolution 을 활용하는 고전적인 모델인 LeNet을 CIFA Image를 이용하여 학습한다. 주어진 이미지를 제대로 분류하는지 성능을 측정한다. Adam Optimazier 활용.

![image](https://user-images.githubusercontent.com/40593455/98448135-6edc1100-216d-11eb-8d03-93169ee244c2.png)

### 4_ResNet

좀더 개선된 모델인 ResNet Model을 구현한다.
![image](https://user-images.githubusercontent.com/40593455/98681309-eac09e00-23a5-11eb-9f2c-cc23f97e86b3.png)

### 5_CGAN

스스로 이미지를 생성해내는 GAN의 개선 버전인 CGAN을 구현한다.

![image](https://user-images.githubusercontent.com/40593455/98448202-b2367f80-216d-11eb-965a-fb3333d1db84.png)

### 6_word2vec

자연어 처리의 기초인 Word2vec에 대하여 학습한다.

### 7_fastText


### 8_CNNforTextSentence

### Final Project : Mask Classifier

OpenCV의 FaceDetection을 사용하여 MaskDetection 모델을 생성한다.

이 때 마스크를 착용한 모델과 착용하지 않은 데이터셋을 생성할 때 GAN을 이용하여 마스크를 쓰지 않은 사람에게 마스크를 착용하여 Data Augmentation을 진행.

이 후 마스크 착용에 대한 모델을 별도로 만든다.
![image](https://user-images.githubusercontent.com/40593455/98681486-1d6a9680-23a6-11eb-8d02-6f2cf28a207a.png)

현재 진행중인 프로젝트 (완료 예정일 12/16)



