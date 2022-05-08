import cv2
import threading
from tracker import vehicle_tracker
from detector import vehicle_detector


def start_detect(input_video):
	counting = 0
        #frame计数

	while(input_video.isOpened()):
		ret, frame = input_video.read()

		if(ret):
			counting += 1
			check_frame_interval = counting % frame_detection_interval

		if(counting == 1 or check_frame_interval == 0):
			detected_frames_dict[counting], detected_frames_count_dict[counting] = detector.detecting(frame)

		else:
			break

def start_traking(start_frame_index):
	counting = start_frame_index
	first_frame = input_video_copy.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
	first_box = detected_frames_dict[start_frame_index]["box_points"]

if __name__ == '__main__':
	#defaul setting
	path = 'traffic-mini.mp4'
	input_video = cv2.VideoCapture(path)
	input_video_copy = input_video.copy()
	frame_detection_interval = 1
	detector = vehicle_detector()
	tracker = vehicle_tracker()

	if (not input_video):
		raise ValueError(
			"No video source")
	else:
		try:
			detected_frames_dict = {} #保存每帧的检测结果，name, percentage_probability, box_points
			detected_frames_count_dict = {}

			frame_width = int(input_video.get(3))
			frame_height = int(input_video.get(4))
			#提取视频长宽

			detect_t = threading.Thread(target=start_detect,args=(input_video,))
			detect_t.start()
			detect_t.join()


		except:
			raise ValueError(
				"An error occured. It may be that your input video is invalid.")


	