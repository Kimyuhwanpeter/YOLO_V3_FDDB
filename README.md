# YOLO_V3_FDDB
* 아직 학습하는중
* 결과는 따로 오늘 밤에 업로드할 예정
* 지금 하고있는 GAN 연구에 접목시키기 위한 작업
* [[0.16308074 0.08024422], [0.03426952 0.02785091], [0.17176707 0.1526203 ], [0.068794   0.05186673], [0.12877259 0.1151205 ], [0.44492477 0.23083241], [0.10429738 0.08574786], [0.21951485 0.10846578], [0.29080772 0.16321221]] 로 FDDB에 대한 anchor를 구했음
* 위 anchor로 설정하고 다시 학습을 해야함 (집에서 이어서 진행)
* 학습 dataset은 약 1,500 장 뿐임
* backbone 은 Resnet-50을 이용하였음

## Epoch 5 (VOC2012 anchors를 사용했을 때)
![img ](https://github.com/Kimyuhwanpeter/YOLO_V3_FDDB/blob/main/500_2.jpg)

## 직접 구한 anchor 박스를 사용했을 때 (직접구한 anchors를 이용했을 때)
![img2](https://github.com/Kimyuhwanpeter/YOLO_V3_FDDB/blob/main/2500_7.jpg)
<br/>


* 결과적으로는 비슷한것같음
