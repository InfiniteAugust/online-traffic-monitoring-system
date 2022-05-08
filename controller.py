import cv2
import math
import multiprocessing

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
		input_video = cv2.VideoCapture(self.path)

		if (not	input_video):
			raise ValueError(
				"No video source")
		else:
			self.detected_frames_dict = multiprocessing.Manager().dict() # [name, percentage_probability, box_points] 
			self.detected_frames_count_dict = multiprocessing.Manager().dict()
			self.tracked_frames_dict = multiprocessing.Manager().dict()
			#线程共享数据

			self.frame_detection_interval = 1
			self.tracking_state = multiprocessing.Value("d",0)
			self.detecting_state = multiprocessing.Value("d",0)
			try:				
				self.frame_width = int(input_video.get(3))
				self.frame_height = int(input_video.get(4))
				self.video_fps = input_video.get(cv2.CAP_PROP_FPS)
				#提取视频长宽
				input_video.release()
			except:
				raise ValueError(
					"An error occured. It may be that your input video is invalid.")


	def detect(self):
		'''
			start to detect, and save the result
		'''
		from detector import vehicle_detector
		detector = vehicle_detector()
		input_video = cv2.VideoCapture(self.path)

		counting = 0
	    #frame计数

		self.detecting_state.value = 1

		while(input_video.isOpened()):
			ret, frame = input_video.read()

			if(ret and self.detecting_state.value == 1):
				counting += 1
				check_frame_interval = counting % self.frame_detection_interval

				if(counting == 1 or check_frame_interval == 0):
					self.detected_frames_dict[counting], self.detected_frames_count_dict[counting] = detector.detecting(frame)
					print(self.detected_frames_dict[counting])
			else:
				continue

			if counting == 5:
				break

		input_video.release()

		self.detecting_state.value = 0

	def start_detect(self):
		detect_process = multiprocessing.Process(target=self.detect)
		'''start the detection process'''
		if self.detecting_state.value == 1:
			self.detecting_state.value = 0
			#stop the old process

		track_process = multiprocessing.Process(target=self.track, args=(1, [200,200]))

		if self.tracking_state.value == 1:
			self.tracking_state.value = 0
			#stop the old process

		track_process.start()
		detect_process.start()
		detect_process.join()
		track_process.join()

	def track(self, start_frame_index, position):
		'''
			start to track, and save the result
			parameters:
			position should be the relative (x, y) on the video
		'''
		from tracker import vehicle_tracker

		related_Items = []

		while len(self.detected_frames_dict) < start_frame_index:
			pass
		# wait for detector

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

			input_video = cv2.VideoCapture(self.path)
			input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
			ok, first_frame = input_video.read()
			if ok:
				self.tracking_state.value = 1

				counting = start_frame_index
				my_tracker = vehicle_tracker() 
				my_tracker.init_tracker(first_frame, first_box)

				while(input_video.isOpened()):
					ret, target_frame = input_video.read()

					if(ret and self.tracking_state.value == 1):
						counting += 1
						success, result = my_tracker.tracking(target_frame)
						#只保存能追踪出来的帧
						if success:
							self.tracked_frames_dict[counting] = result
							print(result)
						else:
							self.tracked_frames_dict[counting] = (0,0,0,0)
						#速度检测 TODO

					else:
						break
					if counting == 5:
						break

			input_video.release()

		self.tracking_state.value = 0

	def start_track(self, start_frame_index, position):
		'''start the tracking processs'''
		track_process = multiprocessing.Process(target=self.track, args=(start_frame_index, position))

		if self.tracking_state.value == 1:
			self.tracking_state.value = 0
			#stop the old process

		track_process.start()
		track_process.join()

	def speed_calculate(self, pre_box, cur_box, distance_coefficient):
		'''calculate the speed of a given object'''
		pre_centre = ((pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2)
		cur_centre = ((cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2)
		pixel_distance = math.sqrt((pre_centre[0] - cur_centre[0]) ** 2 + (pre_centre[1] - cur_centre[1]) ** 2)
		speed = pixel_distance * distance_coefficient * self.video_fps * 3.6 # expressed in km/h

		return speed

if __name__ == '__main__':
	controller = Controller()
	# controller.start_detect()
	controller.start_detect()
	
	