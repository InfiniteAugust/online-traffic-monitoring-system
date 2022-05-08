import os
import cv2
import colorsys
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
from keras.layers import Input

from imageai.Detection.YOLOv3.models import yolo_main, tiny_yolo_main
from imageai.Detection.YOLOv3.utils import letterbox_image, yolo_eval
from imageai.Detection.keras_retinanet.utils.colors import label_color

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class vehicle_detector():
    """接收图片，并保存处理结果"""
    def __init__(self):
        super(vehicle_detector, self).__init__()
        self.modelPath = 'yolo.h5'
        self.__model_collection = []

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                                   79: 'toothbrush'}

        hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                      for x in range(len(self.numbers_to_names))]
        #通过HSV颜色空间为每个类型分配颜色

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                           self.colors))
        #转换到RGB颜色空间
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        #打乱颜色
        np.random.seed(None)
        
        self.__yolo_iou = 0.5 #交并比设置
        self.__yolo_score = 0.1 
        self.__yolo_anchors = np.array([[ 10. , 13.], [ 16. , 30.], [ 33. , 23.], [ 30. , 61.], [ 62. , 45.], [ 59. , 119.], [116. , 90.], [156. , 198.], [373. , 326.]])
        #anchor box 需要重新计算
        self.__yolo_model_image_size = (416,416)
        #size越大，速度越慢
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.sess = K.get_session()

        model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3, len(self.numbers_to_names))
        model.load_weights(self.modelPath)

        self.__yolo_input_image_shape = K.placeholder(shape=(2,))
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output, self.__yolo_anchors,
                                                                                len(self.numbers_to_names), self.__yolo_input_image_shape,
                                                                                 score_threshold=self.__yolo_score, iou_threshold=self.__yolo_iou)
        #初始化YOLO3

        self.__model_collection.append(model)


    def CustomObjects(self, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False, stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False, cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False, frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False, surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False, banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donot=False, cake=False, chair=False, couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False, keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False, scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donot, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donot", "cake", "chair", "couch", "potted plant", "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if(input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict


    def detecting(self, source_frame, custom_objects=None, minimum_percentage_probability=50):
        '''检测图片，并保存边框'''
        custom_objects = self.CustomObjects(truck=True, bus=True, car=True, bicycle=True, motorcycle=True)

        model = self.__model_collection[0]
        output_objects_array = []

        frame = Image.fromarray(np.uint8(source_frame)) 
        #将数组转换为image对象
        
        new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                          self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))
        #修改视频尺寸, 以满足网络输入

        boxed_image = letterbox_image(frame, new_image_size)
        #缩放图片尺寸到(416，416)
        image_data = np.array(boxed_image, dtype="float32")

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
    
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
            feed_dict={
                model.input: image_data,
                self.__yolo_input_image_shape: [frame.size[1], frame.size[0]],
                K.learning_phase(): 0
            })
        #运行detection

        min_probability = minimum_percentage_probability / 100

        for a, b in reversed(list(enumerate(out_classes))):
            predicted_class = self.numbers_to_names[b]
            #预测的类
            box = out_boxes[a]
            #预测的边框
            score = out_scores[a]
            #预测得分
            if score < min_probability:
                continue

            if (custom_objects != None):
                if (custom_objects[predicted_class] == "invalid"):
                    continue

            label = "{} {:.2f}".format(predicted_class, score)
            #输出信息格式为 预测类，得分

            top, left, bottom, right = box 
            #边框四个顶点
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

            try:
                color = label_color(b)
                #边框颜色, 从label_color中获取
            except:
                color = (255, 0, 0)

            detection_details = (left, top, right, bottom) #(x1, y1, x2, y2)

            each_object_details = {}
            each_object_details["name"] = predicted_class
            each_object_details["percentage_probability"] = score * 100
            each_object_details["box_points"] = detection_details
            output_objects_array.append(each_object_details)
            #每一个object由name percentage_probability box_points组成

        output_objects_count = {}
        for eachItem in output_objects_array:
            #计算每帧每类的数量
            eachItemName = eachItem["name"]
            try:
                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
            except:
                output_objects_count[eachItemName] = 1

        return output_objects_array, output_objects_count
