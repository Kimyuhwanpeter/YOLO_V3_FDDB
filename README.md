# YOLO_V3_FDDB
* ***지금 하고있는 GAN 연구에 접목시키기 위한 작업***
* get_anchor.py에서 k-means clustering을 하여 FDDB의 anchor를 구했음
* 학습 dataset은 약 1,500 장 뿐임 (학습수가 너무적고, 사실 객체에 대한 라벨 클래스를 고려하지 않은것임(바운딩좌표만 집중) )
* backbone 은 Resnet-50을 이용하였음 (Darknet-53의 성능이 일반적으로 좀 더 좋다고 판단됨. ***Receptive field***에 대해 좀 더 집중해야할듯--> GAN도 그렇고 Segmentation에서도 중요한 문제이기 떄문에 detection도 이들과 비슷한점이 많음 )

## Epoch 5 (VOC2012 anchors를 사용했을 때)
![img ](https://github.com/Kimyuhwanpeter/YOLO_V3_FDDB/blob/main/500_2.jpg)

## 직접 구한 anchor 박스를 사용했을 때 (직접구한 anchors를 이용했을 때)
![img2](https://github.com/Kimyuhwanpeter/YOLO_V3_FDDB/blob/main/2500_7.jpg)
<br/>


* 결과적으로는 비슷한것같음
* 라벨 라벨 파일을 잘못 변환시켰음(그래서 얼굴이 제대로 나오지 않고 있는것임)
* Backbone 모델을 resnet-50 말고, ***Receptive field*** 를 강조 시킬수있는 (dilated conv와 같은) 모델을 쓰거나 혹은 Backbone은 기존것을 그대로 사용하되 branch로 설정되어있는 3개의 출력단들의 layer의 구성에 대해 ***Receptive field*** 를 강조시킬 수 있도록 할것 (코딩은 금방하지만, 해봐야할 실험 케이스가 많음)
