"""
Main function to implement Computer Pointer Controller project
"""

import time
import os
import cv2
import sys
import pyautogui

import numpy as np
import logging as log

from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetect
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation


def build_argparser():
	"""
	Parse command line arguments.

	:return: command line arguments
	"""
	
	fd_help = "Path to a trained Face Detection model."
	hp_help = "Path to a trained Head Post Estimation model."
	fl_help = "Path to a trained Facial Landmarks Detection model."
	ge_help = "Path to a trained Gaze Estimation model."
	i_help = "Path to image/video file or 'CAM' for webcam input"
	l_help = "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels impl."
	d_help = "Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Sample will look for a suitable plug-in for device specified (CPU by default)"
	pt_help = "Probability threshold for detections filtering (0.5 by default)"
	mp_help = "Set the precision for mouse movement: high, low, medium."
	ms_help = "Set the speed for mouse movement: fast, slow, medium."
	dis_help = "Flag to display intermediate model outputs."
	np_help = "Flag for not to move the mouse pointer for the gaze estimation."
	
	parser = ArgumentParser()
	
	parser.add_argument("-fd", "--fd_model", required=True, type=str, help=fd_help)
	parser.add_argument("-hp", "--hp_model", required=True, type=str, help=hp_help)
	parser.add_argument("-fl", "--fl_model", required=True, type=str, help=fl_help)
	parser.add_argument("-ge", "--ge_model", required=True, type=str, help=ge_help)
	parser.add_argument("-i", "--input", required=True, type=str, help=i_help)
	parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None, help=l_help)
	parser.add_argument("-d", "--device", type=str, default="CPU", help=d_help)
	parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help=pt_help)
	parser.add_argument("-mp", "--mouse_precision", required=False, type=str, default="high", help=mp_help)
	parser.add_argument("-ms", "--mouse_speed", required=False, type=str, default="fast", help=ms_help)
	parser.add_argument("-dis", "--display", required=False, action='store_true', help=dis_help)
	parser.add_argument("-np", "--no_pointer_move", required=False, action='store_true', help=np_help)
	
	return parser

def convert_time(n):
	return time.strftime("%H:%M:%S", time.gmtime(n))
	
def handle_input_stream(input_stream):
	"""
	Handle input image, video or CAM.

	:param input_stream: path of video/image file or 'CAM' string for webcam
	:return: input_type, input_path
	"""
	
	# Checks if webcame return input_type "cam" and input_path as None
	if input_stream == 'CAM':
		return 'cam', None
	
	else:
		file_extension = input_stream.split(".")[-1]
		
		# Checks if image
		if file_extension in ['jpg', 'jpeg', 'bmp', 'png']:
			return 'image', input_stream
		# Checks if video
		elif file_extension in ['avi', 'mp4']:
			return 'video', input_stream
		else:
			log.error("Unsupported file Extension. Allowed extensions are ['jpg', 'jpeg', 'bmp', 'png', 'avi', 'mp4']")
			sys.exit(1)

def infer_on_stream(args):
	"""
	Initialize the inference network, process input video stream or image,
	and output stats and video.

	:param args: Command line arguments parsed by `build_argparser()`
	:return: None
	"""
	# Handle input stream
	input_type, input_path = handle_input_stream(args.input)
	
	### Load the models ###
	start_model_load_time = time.time()
	
	# Face Detection Model
	face_detection_model = FaceDetect(args.fd_model, args.device, args.prob_threshold, args.cpu_extension)
	face_detection_model.load_model()
	
	# Head Pose Estimation Model
	head_pose_model = HeadPoseEstimation(args.hp_model, args.device, args.cpu_extension)
	head_pose_model.load_model()
	
	# Facial Landmarks Detection Model
	facial_landmarks_model = FacialLandmarksDetection(args.fl_model, args.device, args.cpu_extension)
	facial_landmarks_model.load_model()
	
	# Gaze Estimation Model
	gaze_estimation_model = GazeEstimation(args.ge_model, args.device, args.cpu_extension)
	gaze_estimation_model.load_model()
	
	total_model_load_time = convert_time(time.time() - start_model_load_time)
	log.warning("Total Model Load time: {}".format(total_model_load_time))
	
	### Check the models ###
	face_detection_model.check_model()
	head_pose_model.check_model()
	facial_landmarks_model.check_model()
	gaze_estimation_model.check_model()
	
	# load Input Feeder
	feed = InputFeeder(input_type=input_type, input_file=input_path)
	feed.load_data()
	
	# Load Mouse Controller 
	mouse_pointer = MouseController(args.mouse_precision, args.mouse_speed)
	
	# Create output file
	w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
	output_video = cv2.VideoWriter(os.path.join('output_video.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h), True)
	
	### Start Inference ###
	start_inference_time=time.time()
	
	for frame_flag, image in feed.next_batch():
		if not frame_flag:
			log.warning("Frame not received. May be end of stream.")
			break
		
		key_pressed = cv2.waitKey(60)
		
		# Face detection
		face_coords = face_detection_model.predict(image)
		
		# Check if the faces are detected in the image
		if not face_coords:
			log.error("Face not detected")
			continue
			
		# Only proceed with the top most detected face
		coords = face_coords[0]   # [xmin, ymin, xmax, ymax] 
		
		# Get the cropped image 
		cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]] 
		
		output_image = image.copy()
		
		if args.display:
			# Draw face bounding box on the image
			cv2.rectangle(output_image, (coords[0], coords[1]), (coords[2], coords[3]), (255, 0, 0), 1)
		
		# Facial landmarks Detection
		output_image, eyes_img, eyes_center = facial_landmarks_model.predict(output_image, cropped_face, coords, args.display)
		
		# Head Pose Estimation
		output_image, head_pose_angles = head_pose_model.predict(output_image, cropped_face, coords, args.display)
		
		# Gaze Estimation
		output_image, gaze_vector = gaze_estimation_model.predict(output_image, eyes_img, eyes_center, head_pose_angles, coords, args.display)
		
		# Show the output image
		cv2.imshow('mouse_controller', output_image)
		
		if not args.no_pointer_move:
			# Move mouse pointer according to the gaze estimation, 
			# if the flag no_pointer_move set to False
			mouse_pointer.move(gaze_vector[0], gaze_vector[1])
		
		# Write to output video
		output_video.write(output_image)
		
		if key_pressed == 27:
			break
		
	total_inference_time = convert_time(time.time() - start_inference_time)
	log.warning("Total Inference time: {}".format(total_inference_time))
	
	# Release the capture and destroy any OpenCV windows
	feed.close()
	cv2.destroyAllWindows()

def main():
	"""
	Get the input arguments and main function for Computer Pointer Controller
	
	:return: None
	"""
	# Get the input arguments
	args = build_argparser().parse_args()
	
	# Perform inference on the input stream
	infer_on_stream(args)

if __name__=='__main__':
	main()