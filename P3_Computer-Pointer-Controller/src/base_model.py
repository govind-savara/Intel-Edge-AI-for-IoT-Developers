'''
This is a base class for a model. This class implements the sample methods for the models.
'''
import cv2
import sys
import logging as log
from openvino.inference_engine import IECore


class BaseModel:
	'''
	Class for the Base Model.
	'''
	def __init__(self, model_name, model_desc, device='CPU', extensions=None):
		'''
		Initializing instance variables.
		'''
		self.model_weights = model_name+'.bin'
		self.model_structure = model_name+'.xml'
		self.model_desc = model_desc
		self.device = device
		self.extension = extensions

		try:
			self.core = IECore()
			self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)
		except Exception as e:
			raise ValueError("Could not Initialize the {} network. Please check the model path.".format(self.model_desc))

		self.input_name = next(iter(self.model.inputs))
		self.input_shape = self.model.inputs[self.input_name].shape
		self.output_name = next(iter(self.model.outputs))
		self.output_shape = self.model.outputs[self.output_name].shape

	def load_model(self):
		'''
		This method is for loading the model to the device specified by the user.
		'''
		# Add CPU extension, if applicable
		if self.device == 'CPU' and self.extension:
			self.core.add_extension(self.extension, self.device)
		
		try:
			self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
		except Exception as e:
			raise ValueError("Unable to load the {} model. Check the model initialization.".format(self.model_desc))

	def predict(self, image):
		'''
		This method is meant for running predictions on the input image.
		'''
		pass

	def check_model(self):
		'''
		Check Model layers.
		'''
		# Supported Layers
		supported_layers = self.core.query_network(network=self.model, device_name=self.device)
		
		# The support for GPU code is taken from the following Intel Community discussion link
		# source https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/Unable-to-configure-GPU-clDNNPlugin-when-using-OpenVINO-Python/td-p/1144533
		if self.device == 'GPU':
			supported_layers.update(self.core.query_network(self.model, 'CPU'))
		
		# Check for Unsupported layers
		unsupported_layers = [layer for layer in self.model.layers.keys() if layer not in supported_layers]
		
		if unsupported_layers:
			log.error("Unsupported layers found {}".format(unsupported_layers))
			sys.exit(1)
		else:
			log.info("Checking {} model is done. All layers are supported".format(self.model_desc))

	def preprocess_input(self, image):
		'''
		Pre-process the input image - resize, transpose and reshape the image as needed
		'''
		dimention = (self.input_shape[3], self.input_shape[2])

		p_frame = cv2.resize(image, dimention)
		p_frame = p_frame.transpose((2,0,1))
		p_frame = p_frame.reshape(1, *p_frame.shape)

		return p_frame

	def preprocess_output(self, outputs):
		'''
		Before feeding the output of this model to the next model,
		we might have to pre-process the output. This function is where we can do that.
		'''
		pass
