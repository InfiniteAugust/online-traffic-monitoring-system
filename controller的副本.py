import cv2
import multiprocessing
import math
from tracker import vehicle_tracker
from detector import vehicle_detector
import json

def box_index(position, box):
	'''give the relative score of box and the position'''
	x = position[0]
	y = position[1]
	x1 = box[0]
	y1 = box[1]
	x2 = box[2]
	y2 = box[3]

	return 1 - (abs(x - (x1+x2) / 2) / abs(x2 - x1) + abs(y - (y1 + y2) / 2) / abs(y1 -y2))

class Controller(object):
	"""controll the whole process"""
	def __init__(self):
		super(Controller, self).__init__()
		self.path = 'traffic-mini.mp4'
		self.input_video = cv2.VideoCapture(self.path)
		self.frame_detection_interval = 1
		self.video_fps = self.input_video.get(cv2.CAP_PROP_FPS)
		self.detector = vehicle_detector()
		self.tracker = vehicle_tracker()

		self.tracking_state = False
		self.detecting_state = False

		if (not self.input_video):
			raise ValueError(
				"No video source")
		else:
			try:
				self.detected_frames_dict = {} #保存每帧的检测结果, name, percentage_probability, box_points
				self.detected_frames_count_dict = {}

				self.tracked_frames_dict = {}

				self.frame_width = int(self.input_video.get(3))
				self.frame_height = int(self.input_video.get(4))
				#提取视频长宽

			except:
				raise ValueError(
					"An error occured. It may be that your input video is invalid.")


	def start_detect(self):
		'''
			start to detect, and save the result
		'''
		self.detecting_state = True

		counting = 0
	        #frame计数

		while(self.input_video.isOpened()):
			ret, frame = self.input_video.read()

			if(ret):
				counting += 1
				check_frame_interval = counting % self.frame_detection_interval

			if(counting == 1 or check_frame_interval == 0):
				self.detected_frames_dict[counting], self.detected_frames_count_dict[counting] = self.detector.detecting(frame)
				print(self.detected_frames_dict[counting])
			else:
				continue

			if counting == 3:
					break

		with open('data.json', 'w') as f:
			json.dump(self.detected_frames_dict, f)

			

	def start_tracking(self, start_frame_index, position):
		'''
			start to track, and save the result
			parameter position should be the relative (x, y) on the video
		'''
		self.tracking_state = True

		counting = start_frame_index
		t_input_video = cv2.VideoCapture(self.path)

		related_Items = []

		for eachItem in self.detected_frames_dict[start_frame_index]:
			eachItemPositon = eachItem["box_points"]
			if position[0] > eachItemPositon[0] and position[0] < eachItemPositon[2] and position[1] < eachItemPositon[3] and position[1] > eachItemPositon[1]:
				related_Items.append(eachItem)
		#selected related boxes clicked by the user

		if related_Items:
			target_item = related_Items[0] #包含了追踪对象的 name, percentage_probability, box_points
			first_box = target_item["box_points"]
			socre = box_index(position, first_box)

			if len(related_Items) > 1:
				for item in related_Items[1:-1]:
					new_score = box_index(position, item["box_points"])
					if new_score > socre:
						target_item = item
						socre = new_score
						first_box = item["box_points"]
			#select out the correct box

			t_input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index) #从特定帧开始追踪
			
			first_frame = t_input_video.read()

			self.tracker.init_tracker(first_frame, first_box)
			
			while(t_input_video.isOpened()):
				ret, target_frame = t_input_video.read()

				if(ret):
					counting += 1
					self.tracked_frames_dict[counting] = self.tracker.tracking(target_frame)
					#只保存能追踪出来的帧

					#first_frame = target_frame
					#first_box = self.tracked_frames_dict[counting]
					#速度检测 TODO

				else:
					continue

	def speed_calculate(self, pre_box, cur_box, distance_coefficient):
		
		pre_centre = ((pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2)
		cur_centre = ((cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2)
		pixel_distance = math.sqrt((pre_centre[0] - cur_centre[0]) ** 2 + (pre_centre[1] - cur_centre[1]) ** 2)
		speed = pixel_distance * distance_coefficient * self.video_fps * 3.6 # expressed in km/h

		return speed

if __name__ == '__main__':
	controller = Controller()
	controller.start_detect()
	