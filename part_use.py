import cv2
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from keras.layers import Input
from PIL import Image
import colorsys

from imageai.Detection.keras_retinanet.models.resnet import resnet50_retinanet
from imageai.Detection.keras_retinanet.utils.image import read_image_bgr, read_image_array, read_image_stream, preprocess_image, resize_image
from imageai.Detection.keras_retinanet.utils.visualization import draw_box, draw_caption
from imageai.Detection.keras_retinanet.utils.colors import label_color

from imageai.Detection.YOLOv3.models import yolo_main, tiny_yolo_main
from imageai.Detection.YOLOv3.utils import letterbox_image, yolo_eval
import multiprocessing


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


class My_ObjectDetection:

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []

        # Instance variables for RetinaNet Model
        self.__input_image_min = 1333
        self.__input_image_max = 800

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

        # Unique instance variables for YOLOv3 and TinyYOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__yolo_anchors = np.array([[ 10. , 13.], [ 16. , 30.], [ 33. , 23.], [ 30. , 61.], [ 62. , 45.], [ 59. , 119.], [116. , 90.], [156. , 198.], [373. , 326.]])
        self.__yolo_model_image_size = (416,416)
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.sess = K.get_session()


        # Unique instance variables for TinyYOLOv3.
        self.__tiny_yolo_anchors = np.array([[ 10. , 14.], [ 23. , 27.], [ 37. , 58.], [ 81. , 82.], [135. , 169.],[344. , 319.]])




    def setModelTypeAsRetinaNet(self):
        """
        'setModelTypeAsRetinaNet()' is used to set the model type to the RetinaNet model
        for the video object detection instance instance object .
        :return:
        """
        self.__modelType = "retinanet"

    def setModelTypeAsYOLOv3(self):
        """
                'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model
                for the video object detection instance instance object .
                :return:
                """

        self.__modelType = "yolov3"

    def setModelTypeAsTinyYOLOv3(self):
        """
                        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model
                        for the video object detection instance instance object .
                        :return:
                        """

        self.__modelType = "tinyyolov3"


    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a RetinaNet
          object detection model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if(self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True



    def loadModel(self, detection_speed="normal"):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.


                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"

                :param detection_speed:
                :return:
        """

        if(self.__modelType == "retinanet"):
            if (detection_speed == "normal"):
                self.__input_image_min = 800
                self.__input_image_max = 1333
            elif (detection_speed == "fast"):
                self.__input_image_min = 400
                self.__input_image_max = 700
            elif (detection_speed == "faster"):
                self.__input_image_min = 300
                self.__input_image_max = 500
            elif (detection_speed == "fastest"):
                self.__input_image_min = 200
                self.__input_image_max = 350
            elif (detection_speed == "flash"):
                self.__input_image_min = 100
                self.__input_image_max = 250
        elif(self.__modelType == "yolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (416,416)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (208, 208)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (128, 128)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (96, 96)

        elif (self.__modelType == "tinyyolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (832, 832)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (576, 576)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (272, 272)


        if (self.__modelLoaded == False):
            if(self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif(self.__modelType == "retinanet"):
                model = resnet50_retinanet(num_classes=80)
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
            elif (self.__modelType == "yolov3"):
                model = yolo_main(Input(shape=(None, None, 3)), len(self.__yolo_anchors) // 3, len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output, self.__yolo_anchors,
                                                                  len(self.numbers_to_names), self.__yolo_input_image_shape,
                                                                  score_threshold=self.__yolo_score, iou_threshold=self.__yolo_iou)


                self.__model_collection.append(model)
                self.__modelLoaded = True

            elif (self.__modelType == "tinyyolov3"):
                model = tiny_yolo_main(Input(shape=(None, None, 3)), len(self.__tiny_yolo_anchors) // 2, len(self.numbers_to_names))
                model.load_weights(self.modelPath)

                hsv_tuples = [(x / len(self.numbers_to_names), 1., 1.)
                              for x in range(len(self.numbers_to_names))]
                self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                self.colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                        self.colors))
                np.random.seed(10101)
                np.random.shuffle(self.colors)
                np.random.seed(None)

                self.__yolo_input_image_shape = K.placeholder(shape=(2,))
                self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = yolo_eval(model.output, self.__tiny_yolo_anchors,
                                                                  len(self.numbers_to_names), self.__yolo_input_image_shape,
                                                                  score_threshold=self.__yolo_score, iou_threshold=self.__yolo_iou)


                self.__model_collection.append(model)
                self.__modelLoaded = True

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

        """
                         The 'CustomObjects()' function allows you to handpick the type of objects you want to detect
                         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
                         which you can easily set to true for any number of objects available.  This function
                         returns a dictionary which must be parsed into the 'detectCustomObjectsFromImage()'. Detecting
                          custom objects only happens when you call the function 'detectCustomObjectsFromImage()'


                        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present

                        :param boolean_values:
                        :return: custom_objects_dict
                """

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

    def detectCustomObjectsFromVideo(self, custom_objects=None, input_file_path="", camera_input = None , output_file_path="", frames_per_second=20, frame_detection_interval=1, minimum_percentage_probability=50, log_progress=False, display_percentage_probability=True, display_object_name = True, save_detected_video = True, per_frame_function = None, per_second_function = None, per_minute_function = None, video_complete_function = None, return_detected_frame = False ):

        if (input_file_path == "" and camera_input == None):
            raise ValueError(
                "You must set 'input_file_path' to a valid video file, or set 'camera_input' to a valid camera")
        elif (save_detected_video == True and output_file_path == ""):
            raise ValueError(
                "You must set 'output_video_filepath' to a valid video file name, in which the detected video will be saved. If you don't intend to save the detected video, set 'save_detected_video=False'")

        else:
            try:
                output_frames_dict = {}
                output_frames_count_dict = {}

                input_video = cv2.VideoCapture(input_file_path)
                if (camera_input != None):
                    input_video = camera_input

                output_video_filepath = output_file_path + '.avi'

                frame_width = int(input_video.get(3))
                frame_height = int(input_video.get(4))
                output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                               frames_per_second,
                                               (frame_width, frame_height))

                counting = 0
                predicted_numbers = None
                scores = None
                detections = None

                model = self.__model_collection[0]

                cv2.namedWindow("Window")
                while (input_video.isOpened()):
                    ret, frame = input_video.read()

                    if (ret == True):

                        output_objects_array = []

                        counting += 1

                        if (log_progress == True):
                            print("Processing Frame : ", str(counting))

                        detected_copy = frame.copy()
                        detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)

                        frame = Image.fromarray(np.uint8(frame)) #将数组转换为图像

                        new_image_size = (self.__yolo_model_image_size[0] - (self.__yolo_model_image_size[0] % 32),
                                          self.__yolo_model_image_size[1] - (self.__yolo_model_image_size[1] % 32))


                        boxed_image = letterbox_image(frame, new_image_size) #letterbox函数从yolov3中导入
                        image_data = np.array(boxed_image, dtype="float32")

                        image_data /= 255.
                        image_data = np.expand_dims(image_data, 0)

                        check_frame_interval = counting % frame_detection_interval

                        if (counting == 1 or check_frame_interval == 0):
                            out_boxes, out_scores, out_classes = self.sess.run(
                                [self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes],
                                feed_dict={
                                    model.input: image_data,
                                    self.__yolo_input_image_shape: [frame.size[1], frame.size[0]],
                                    K.learning_phase(): 0
                                })
                        #将数据集输入到模型，并得到结果，需要修改

                        min_probability = minimum_percentage_probability / 100

                        for a, b in reversed(list(enumerate(out_classes))):
                            predicted_class = self.numbers_to_names[b]
                            box = out_boxes[a]
                            score = out_scores[a]

                            if score < min_probability:
                                continue


                            if (custom_objects != None):
                                if (custom_objects[predicted_class] == "invalid"):
                                    continue

                            label = "{} {:.2f}".format(predicted_class, score)

                            top, left, bottom, right = box 
                            #边框四个顶点
                            top = max(0, np.floor(top + 0.5).astype('int32'))
                            left = max(0, np.floor(left + 0.5).astype('int32'))
                            bottom = min(frame.size[1], np.floor(bottom + 0.5).astype('int32'))
                            right = min(frame.size[0], np.floor(right + 0.5).astype('int32'))

                            try:
                                color = label_color(b)
                            except:
                                color = (255, 0, 0)

                            detection_details = (left, top, right, bottom)
                            draw_box(detected_copy, detection_details, color=color)
                            #画边框

                            if (display_object_name == True and display_percentage_probability == True):
                                draw_caption(detected_copy, detection_details, label)
                            elif (display_object_name == True):
                                draw_caption(detected_copy, detection_details, predicted_class)
                            elif (display_percentage_probability == True):
                                draw_caption(detected_copy, detection_details, str(score * 100))
                            #添加额外信息

                            each_object_details = {}
                            each_object_details["name"] = predicted_class
                            each_object_details["percentage_probability"] = score * 100
                            each_object_details["box_points"] = detection_details
                            output_objects_array.append(each_object_details)
                            #每一个object由name percentage_probability box_points组成

                        #for循环结束,当前帧处理完成
                        
                        output_frames_dict[counting] = output_objects_array

                        output_objects_count = {}
                        for eachItem in output_objects_array:
                            eachItemName = eachItem["name"]
                            try:
                                output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                            except:
                                output_objects_count[eachItemName] = 1

                        output_frames_count_dict[counting] = output_objects_count

                        detected_copy = cv2.cvtColor(detected_copy, cv2.COLOR_BGR2RGB)
                        #deteced_copy为当前帧处理好的图片

                        if (save_detected_video == True):
                            output_video.write(detected_copy)
                        
                        cv2.imshow("Window", detected_copy)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        #将当前帧显示到窗口


                    else:
                        break

                    #while循环结束，视频处理完成
                input_video.release()
                output_video.release()
                cv2.destroyAllWindows()
                
            except:
                raise ValueError(
                    "An error occured. It may be that your input video is invalid. Ensure you specified a proper string value for 'output_file_path' is 'save_detected_video' is not False. "
                    "Also ensure your per_frame, per_second, per_minute or video_complete_analysis function is properly configured to receive the right parameters. ")

def demo():

  path = os.getcwd()

  detector = My_ObjectDetection()
  detector.setModelTypeAsYOLOv3()
  detector.setModelPath(os.path.join(path, 'yolo.h5'))
  detector.loadModel()

  custom_objects = detector.CustomObjects(truck=True, bus=True, car=True, bicycle=True, motorcycle=True)
  detector.detectCustomObjectsFromVideo(custom_objects=custom_objects,
                                        input_file_path=os.path.join(path, "traffic-mini.mp4"),
                                        save_detected_video=False,
                                        frames_per_second=20, log_progress=True)


demo()
