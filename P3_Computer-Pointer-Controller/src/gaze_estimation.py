'''
This is a class implementation for Gaze Estimation model.
'''
import cv2
import logging as log
from base_model import BaseModel

class GazeEstimation(BaseModel):
	'''
	Class for the Gaze Estimation Model.
	'''
	def __init__(self, model_name, device='CPU', extensions=None):
		'''
		Initializing instance variables.
		'''
		super().__init__(model_name, 'Gaze Estimation', device, extensions)

		# Get the eye image shape
		self.input_shape = self.model.inputs['left_eye_image'].shape
		# print('Gaze estimation input shape: {}, {}'.format(self.input_shape, self.model.inputs))

	def predict(self, image, eyes_img, eyes_center, head_pose_angles, coords, display):
		'''
		This method is meant for running predictions on the eye images to get gaze estimation.
		'''
		# Pre-process the input eye images
		left_eye_frame = self.preprocess_input(eyes_img[0])
		right_eye_frame = self.preprocess_input(eyes_img[1])
		
		# model input arguments
		input_arg = {'left_eye_image': left_eye_frame,
					 'right_eye_image': right_eye_frame,
					 'head_pose_angles': head_pose_angles
					}
		
		try:
			# Start asynchronous inference for specified request
			infer_request = self.network.start_async(request_id=0, inputs=input_arg)
		except Exception as e:
			raise ValueError("Unable to start the asynchronous inference. Restart the {} model.".format(self.model_desc))
		
		# Wait for the inference result
		if infer_request.wait() == 0:
			# Get the result of the inference request
			infer_result = infer_request.outputs[self.output_name]
			
			log.info("{} model output: {}".format(self.model_desc, infer_result))
			
			# pre-process gaze estimation and draw the axis
			output_image, gaze_vector = self.preprocess_output(infer_result, eyes_center, image, coords, display)
			
		return output_image, gaze_vector


	def preprocess_output(self, outputs, eyes_center, image, coords, display):
		'''
		Pre-process the output to get the gaze vector.
		'''
		gaze_vector = outputs[0]
		x = gaze_vector[0]
		y = gaze_vector[1]
		z = gaze_vector[2]
		
		xmin, ymin, xmax, ymax = coords

		# Left eye center
		left_eye_center_x = int(eyes_center[0][0] + xmin)
		left_eye_center_y = int(eyes_center[0][1] + ymin)
			
		# Right eye center
		right_eye_center_x = int(eyes_center[1][0] + xmin)
		right_eye_center_y = int(eyes_center[1][1] + ymin)
		
		if display:
			# Drawing gaze axis on the image
			cv2.arrowedLine(image, (left_eye_center_x, left_eye_center_y), (left_eye_center_x + int(x * 100), left_eye_center_y + int(-y * 100)), (255,50,100), 5)
			cv2.arrowedLine(image, (right_eye_center_x, right_eye_center_y), (right_eye_center_x + int(x * 100), right_eye_center_y + int(-y * 100)), (255,50,100), 5)
			
		log.info("Gaze Vector: {}".format(gaze_vector))

		return image, gaze_vector
