# Object detection model 사용법

The anboims project used the mask-rcnn and faster-rcnn models, referring to the mmdetection guidelines.

The IDCard dataset will use the mask-rcnn model, and the signboard dataset will use the faster-rcnn model.

The first thing you need to do is set up your environment. In general, it is recommended to use the Anaconda virtual environment, and you can set up the virtual environment by referring to the link.

## Train with customized datasets
1. Prepare the customized dataset
2. Prepare a config
3. Train, test, execution

#### Prepare the customized dataset
Since roboflow supports the coco-mmdetection format, we recommend using that platform to label the dataset.
https://roboflow.com/

#### Preapre a config
The second step is to preapre a config. If you would like to refer to the config file of our project, please refer to the mmdetection/configs/anboims_dataset files.

#### Train, test, execution
1. IDCard (mask -rcnn)

Train a new model
```
python tools/train.py configs/anboims_dataset/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py
```

Test and inference
```
python tools/test.py configs/anboims_dataset/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset
```

Excution
By entering this code, you can check the 4-point coordinates of the ID card image and the output image.
```
python idcard_output.py
```

### Details about execution code
Code: idcard_output.py 
In fact, it is not ture that only idcard_output.py is needed. In other words, we need other python file code to execute this file.

1. config_file 
=> mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset
- base: mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py
    base: mask-rcnn_r50_fpn_1x_coco.py
        base: mask-rcnn_r50_fpn.py  coco_instance.py    schedule_1x.py  default_runtime.py
        


- dataset root