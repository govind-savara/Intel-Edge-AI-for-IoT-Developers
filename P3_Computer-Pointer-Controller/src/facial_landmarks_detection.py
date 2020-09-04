'''
This is a class implementation for Facial Landmarks Detection model.
'''
import cv2
import logging as log
from base_model import BaseModel

class FacialLandmarksDetection(BaseModel):
	'''
	Class for the Facial Landmarks Detection Model.
	'''
	def __init__(self, model_name, device='CPU', extensions=None):
		'''
		Initializing instance variables.
		'''
		super().__init__(model_name, 'Facial Landmarks Detection', device, extensions)

	def predict(self, image, face, face_coords, display):
		'''
		This method is meant for running predictions on the input face image to get the Facial landmarks.
		'''
		# Pre-process the input face image
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
			infer_result = infer_request.outputs[self.output_name]
			
			log.info("{} model output: {}".format(self.model_desc, infer_result))
			
			# Get the output image with bounding boxes and eyes center points
			output_image, eyes_img, eyes_center = self.preprocess_output(infer_result, face_coords, image, display)
			
		return output_image, eyes_img, eyes_center

	def preprocess_output(self, outputs, face_coords, image, display):
		'''
		Before feeding the output of this model to the next model,
		pre-processing the output.
		'''
		# Get the landmarks from the outputs
		landmarks = outputs.reshape(1, 10)[0]

		# Get the height and width of the face image
		# face_coords = (xmin,ymin,xmax,ymax)
		height = face_coords[3] - face_coords[1] # ymax - ymin
		width = face_coords[2] - face_coords[0]  # xmax - xmin

		eyes = [] # list to hold left and right eye images

		# Draw bounding boxes for eyes
		for i in range(2):
			x = int(landmarks[i*2] * width)
			y = int(landmarks[i*2+1] * height)
			
			if display:
				# Drawing the box in the image
				# considering offset of face from main image
				cv2.circle(image, (face_coords[0]+x, face_coords[1]+y), 30, (0,255,i*255), 2)
			
			# bounding box co-ordinates of eye
			xmin = face_coords[0] + x - 30
			ymin = face_coords[1] + y - 30
			xmax = face_coords[0] + x + 30
			ymax = face_coords[1] + y + 30
			
			eyes.append(image[ymin:ymax, xmin:xmax].copy())
		
		# eye centers
		left_eye_center =[landmarks[0] * width, landmarks[1] * height]
		right_eye_center = [landmarks[2] * width, landmarks[3] * height]
		
		log.info("Left eye center: {}, Right eye center: {}".format(left_eye_center, right_eye_center))
		
		return image, eyes, [left_eye_center, right_eye_center]