import cv2
import math
import multiprocessing

def test():
    my_tracker = cv2.TrackerKCF_create()

def test2():
    from detector import vehicle_detector 


if __name__ == '__main__':
    
    # controller.start_detect()
    p = multiprocessing.Process(target=test)
    p2 = multiprocessing.Process(target=test2)
    p2.start()
    p2.join()
    p.start()
    p.join()
    