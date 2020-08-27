"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from accuracy import find_accuracy

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def handle_input_stream(input_stream):
    """
    Handle input image, video or CAM. 
    """
    is_image = False
    
    # Checks if webcame return 0 for opencv to read camera feed
    if input_stream == 'CAM':
        return is_image, 0
    # Checks if image
    elif input_stream.endswith('.jpg') or input_stream.endswith('.bmp') or input_stream.endswith('.png') :
        is_image = True
        return is_image, input_stream
    # Otherwise, consider it as video file
    else:
        return is_image, input_stream

def draw_boxes(frame, result, threshold, width, height):
    """
    Draw bounding boxes onto the frame.
    """
    count = 0
    for box in result[0][0]:
        conf = box[2]
        class_id = int(box[1])
        
        # Check if confidence is higher than the threshold
        if conf >= threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            
            # Draw box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            
            # Increment People count
            count += 1
                
    return frame, count

def convert_time(n):
    return time.strftime("%H:%M:%S", time.gmtime(n))

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
        
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Initialize the variables
    last_count = 0
    total_count = 0
    start_time = 0
    detected = False
    last_six_count = []
    frame_num = 0
    
    # List to hold current_count values to calculate the accuracy
    detection_list = []

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    input_shape = infer_network.get_input_shape()
    
    # log.warning("input shape: {}".format(input_shape))

    ### TODO: Handle the input stream ###
    single_image_mode, input_stream = handle_input_stream(args.input)
    
    # Get and open video capture
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Create a video writer for the output video
    if not single_image_mode:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out_people_count_video.mp4', 0x00000021, 30, (width, height))
    else:
        out = None

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)
        
        # Increment Frame number
        frame_num += 1

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        # start time of inference
        inf_start = time.time()

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            inference_time = time.time() - inf_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count = draw_boxes(frame, result, prob_threshold, width, height)
            
            # Append the detected counts to the lists
            last_six_count.append(current_count)
            detection_list.append(current_count)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            inf_time_message = "Inference time: {:.3f}ms"\
                                .format(inference_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            client.publish(inf_time_message)
            
            client.publish("person", json.dumps({"count": current_count}))
            
            # Calculate the total count
            if current_count > last_count and detected == False:
                start_time = frame_num
                detected = True
                total_count += current_count - last_count
                
                ### Topic "person": keys of "total" ###
                client.publish("person", json.dumps({"total": total_count}))
                
            ### Topic "person/duration": key of "duration" ###
            if current_count == 0:
   
                # Check if a person is detected in the current frame and no person was detected in the last five frames
                if (detected and all(x == 0 for x in last_six_count[-5:])):
                    detected = False 
                    
                    # Check if there was a person detected before the last five frames 
                    if(last_six_count[-6] == 1):
                    
                        # Substract the start_time and the last five frames from the current frame_num
                        end_time = frame_num - start_time - 5
                        
                        # Divide end_time by 24 to convert it to seconds, and round it to two decimal places
                        # FPS = 24
                        duration = round(end_time/24, 2)
                        
                        ### Topic "person/duration": key of "duration" ### 
                        client.publish("person/duration", json.dumps({"duration": duration}))
                    else:
                        pass
            
            del last_six_count[:-6]
            last_count = current_count

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.putText(out_frame, "Current Count: {}".format(current_count), (10, height - ((1 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 10, 10), 2)
            cv2.imwrite('output_image.jpg', out_frame)
            
        if key_pressed == 27:
            break
              
    # Print accuracy of the model to the console
    log.warning("Accuracy: {:.2f}%".format(find_accuracy(detection_list)))
    # Release the capture
    cap.release()
    
    # Release the output
    out.release()
    
    # Destroy any OpenCV windows
    cv2.destroyAllWindows()
    
    # Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    
    # Perform inference on the input stream
    start_time = time.time()
    infer_on_stream(args, client)
    end_time = time.time()
    duration = convert_time(end_time - start_time)
    
    log.warning("Inference time: {}".format(duration))


if __name__ == '__main__':
    main()
