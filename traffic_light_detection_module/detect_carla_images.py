import sys
import os
sys.path.insert(0,'traffic_light_detection_module')
import cv2
import json
from postprocessing import draw_boxes
from postprocessing import decode_netout
from predict import predict_with_model_from_file, get_model_from_file,predict_with_model_from_carla
file_path = 'traffic_light_detection_module/config.json'
with open(file_path) as config_buffer:
    config = json.loads(config_buffer.read())


def detect_on_carla_image(model,carla_image):
    netout = predict_with_model_from_carla(config, model, carla_image)

    #plt_image = draw_boxes(carla_image, netout, config['model']['classes'])
    #cv2.imshow('detected_image', plt_image)
    #cv2.waitKey(1)

    list=[]

    image_h, image_w, _ = carla_image.shape
    labels=config['model']['classes']
    for box in netout:
        if box.xmin > image_w or box.xmax > image_w or box.ymin > image_h or box.ymax > image_h:
            continue

        delta_x=((box.xmax*image_w)-(box.xmin*image_w))/2
        delta_y=((box.ymax*image_h)-(box.ymin*image_h))/2
        label = labels[box.get_label()]
        score=round(box.get_score(),4)
        center_x=int((box.xmin*image_w)+delta_x)
        center_y=int((box.ymin*image_h)+delta_y)
        list.append([label,score,center_x,center_y])

    return list
