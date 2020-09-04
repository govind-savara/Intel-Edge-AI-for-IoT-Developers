'''
This is a class implementation for a Head Pose Detection model.
'''
import cv2
import logging as log
from base_model import BaseModel
from math import cos, sin, pi


class HeadPoseEstimation(BaseModel):
	'''
	Class for the Head Pose Estimation Model.
	'''
	def __init__(self, model_name, device='CPU', extensions=None):
		'''
		Initializing instance variables.
		'''
		super().__init__(model_name, 'Head Pose Estimation', device, extensions)

	def predict(self, image, face, face_coords, display):
		'''
		This method is meant for running predictions on the face image to get Head pose estimation.
		'''
		# Pre-process the input image
		frame = self.preprocess_input(face)

		input_arg = {self.input_name: frame}

		try:
			# Start asynchronous inference for specified request
			infer_request = self.network.start_async(request_id=0, inputs=input_arg)
		except Exception as e:
			raise ValueError("Unable to start the asynchronous inference. Restart the {} model.".format(self.model_desc))
		
		# Wait for the inference result
		if infer_request.wait() == 0:
			# Get the result of the inference request
			infer_result = infer_request.outputs
			
			log.info("{} model output: {}".format(self.model_desc, infer_result))
			
			output_image, head_pose_angles = self.preprocess_output(infer_result, face_coords, image, display)
			
		return output_image, head_pose_angles

	def preprocess_output(self, outputs, face_coords, image, display):
		'''
		Pre-process the output to find out the head pose estimation.

		Output layer names in Inference Engine format:
		name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
		name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
		name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
		'''

		yaw = outputs['angle_y_fc'][0][0]
		pitch = outputs['angle_p_fc'][0][0]
		roll = outputs['angle_r_fc'][0][0]
		
		if display:
			# if display flag is set, draw the head pose axes
			image = self.draw_axes(image, face_coords, yaw, pitch, roll)

		return image, [yaw, pitch, roll]
		
	def draw_axes(self, image, face_coords, yaw, pitch, roll):
		'''
		Draw axes for the head pose estimation.
		
		Code source: https://sudonull.com/post/6484-Intel-OpenVINO-on-Raspberry-Pi-2018-harvest
		'''
		
		x_min, y_min, x_max, y_max = face_coords
		
		cos_r = cos(roll * pi / 180)
		sin_r = sin(roll * pi / 180)
		sin_y = sin(yaw * pi / 180)
		cos_y = cos(yaw * pi / 180)
		sin_p = sin(pitch * pi / 180)
		cos_p = cos(pitch * pi / 180)
		
		x = (x_min + x_max) // 2
		y = (y_min + y_max) // 2
		
		# Center to right
		cv2.line(image, (x,y), (x+int(70*(cos_r*cos_y+sin_y*sin_p*sin_r)), y+int(70*cos_p*sin_r)), (0, 0, 255), thickness=3)
		# Center to top
		cv2.line(image, (x, y), (x+int(70*(cos_r*sin_y*sin_p+cos_y*sin_r)), y-int(70*cos_p*cos_r)), (0, 255, 0), thickness=3)
		# Center to forward
		cv2.line(image, (x, y), (x + int(70*sin_y*cos_p), y + int(70*sin_p)), (255, 0, 0), thickness=3)
		
		return image