import sys
import os
sys.path.insert(0,'traffic_light_detection_module')
import cv2
import json
from postprocessing import draw_boxes
from postprocessing import decode_netout
from predict import predict_with_model_from_file, get_model_from_file,predict_with_model_from_carla

# Loading model
file_path = 'traffic_light_detection_module/config.json'
with open(file_path) as config_buffer:
    config = json.loads(config_buffer.read())


def detect_on_carla_image(model,carla_image):

    # taking the predicted output by the model
    netout = predict_with_model_from_carla(config, model, carla_image)
    '''
        Uncomment from row 23 to row 26 if you want to see the bouding boxes around the traffic-ligth in real time
    '''
    #plt_image = draw_boxes(carla_image, netout, config['model']['classes'])
    #resized= cv2.resize(plt_image,(400,400))
    #cv2.imshow('detected_image', resized)
    #cv2.waitKey(1)

    list=[]

    #taking the shape of the Carla image
    image_h, image_w, _ = carla_image.shape

    # taking the labels by the configuration file
    labels=config['model']['classes']

    # for each bounding box, we check if their shape is within the limits of the shape of the Carla image
    for box in netout:
        if box.xmin > image_w or box.xmax > image_w or box.ymin > image_h or box.ymax > image_h:
            continue
        
        # taking the half part of the x-axis and y-axis
        delta_x=((box.xmax*image_w)-(box.xmin*image_w))/2
        delta_y=((box.ymax*image_h)-(box.ymin*image_h))/2

        # taking the label assigned to the bounding box
        label = labels[box.get_label()]
        score=round(box.get_score(),4)

        # computing the centre of x-axis and y-axis
        center_x=int((box.xmin*image_w)+delta_x)
        center_y=int((box.ymin*image_h)+delta_y)

        # output list that contains: predicted label, predicted score and centre along x-axis and y-axis of the bounding box
        list.append([label,score,center_x,center_y])

    return list
