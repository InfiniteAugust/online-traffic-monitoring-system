import cv2

class vehicle_tracker(object):
    """docstring for vehicle_tracker"""
    def __init__(self):
        '''
            初始化tracker
            追踪器默认为KCF
            边框颜色默认为红
        '''
        super(vehicle_tracker, self).__init__()
        self.tracker = cv2.TrackerKCF_create()

    def init_tracker(self, first_frame, bounding_box):
        self.tracker.init(first_frame, bounding_box)

    def tracking(self, target_frame):
        '''
           追踪物体,并返回边框
           first_frame为第一帧的帧数号
           bounding_box为需要追踪物体的边框
           格式为(x1,y1,x2,y2)
           targer_frame: frame that wanted to be updated
        '''

        return self.tracker.update(target_frame)       
        