import cv2
import numpy as np


class vehicle_tracker(object):
    """docstring for vehicle_tracker"""
    def __init__(self, video):
        '''
            初始化tracker
            追踪器默认为KCF
            边框颜色默认为红
        '''
        super(vehicle_tracker, self).__init__()
        self.video = video
        self.tracker = cv2.TrackerKCF_create()
        #设置框的颜色

    def tracking(self, first_frame_index, bounding_box):
        '''
           追踪物体,并保存边框
           first_frame为第一帧的帧数号
           bounding_box为需要追踪物体的边框
           格式为(x1,y1,x2,y2)
        '''
        all_boxes = []
        first_frame = self.video.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)
        count = first_frame_index
        bbox = bounding_box.copy()
        self.tracker.init(first_frame, bbox)

        while self.video.isOpened():          
            #pass
            
            #Start timer
            timer = cv2.getTickCount()

            ok, bbox = tracker.update(frame)

            #Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            if ok:
            #Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                #first point
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                #sencond point
                all_boxes.append(bbox)

            count += 1
            #if not success, ignore this frame

        return all_boxes

    def update(self, new_frame, bounding_box):
        '''
            update when the target is changed
        '''
        self.tracking(new_frame, bounding_box)
        