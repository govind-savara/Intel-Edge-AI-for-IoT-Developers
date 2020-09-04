'''
This is a class implementation for a Face Detection model.
'''
import logging as log
from base_model import BaseModel


class FaceDetect(BaseModel):
	'''
	Class for the Face Detection Model.
	'''
	def __init__(self, model_name, device='CPU', threshold=0.6, extensions=None):
		'''
		Initializing instance variables.
		'''
		super().__init__(model_name, 'Face Detection', device, extensions)
		
		self.threshold = threshold

	def predict(self, image):
		'''
		This method is meant for running predictions on the input image.
		'''
		# Pre-process the input image
		frame = self.preprocess_input(image)

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

			# Get the coordinates of bounding boxes
			face_coords = self.preprocess_output(infer_result, image.shape[0], image.shape[1])
		
		return face_coords

	def preprocess_output(self, outputs, height, width):
		'''
		Pre-process the inference output to get the bounding box coordinates
		'''
		detected_coords = []
		
		for box in outputs[0][0]:
			confidence = box[2]
			
			# Get the box coordinates only if confidence is higher than the threshold
			if confidence >= self.threshold:
				xmin = int(box[3] * width)
				ymin = int(box[4] * height)
				xmax = int(box[5] * width)
				ymax = int(box[6] * height)
				
				detected_coords.append([xmin, ymin, xmax, ymax])
				
		log.info("Face Bounding Box coordinates with higher threshold: {}".format(detected_coords)) 
			
		return detected_coords
