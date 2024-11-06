from mmdet.apis import DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')

# Perform inference
inferencer('demo\\20301159_10.jpg', show=True)
