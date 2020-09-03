
import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys


class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you entered the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        Load the model
        '''
        try:
            self.core = IECore()
            self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        except Exception as e:
            raise ValueError("Unable to load the model. Check the model initialization.")
        
    def predict(self, image):
        '''
        Predict the output of the image using the model
        '''
        # Pre-process the input image
        frame = self.preprocess_input(image)
        
        input_arg = {self.input_name: frame}
        
        try:
            # Start asynchronous inference for specified request
            infer_request = self.network.start_async(request_id=0, inputs=input_arg)
        except Exception as e:
            raise ValueError("Unable to start the asynchronous inference. Restart the network model.")
        
        # Wait for the inference result
        if infer_request.wait() == 0:
            # Get the result of the inference request
            infer_result = infer_request.outputs[self.output_name]
            
            # Get the coordinates of bounding boxes
            box_coords = self.preprocess_outputs(infer_result)
            
            # draw bounding boxes of output in the image
            image_output, detected_list = self.draw_outputs(image, box_coords)
            
        return detected_list, image_output
    
    def draw_outputs(self, image, coordinates):
        '''
        Draw bounding boxes in the image using the coordinates
        '''
        width = image.shape[1]
        height = image.shape[0]
        updated_coords = []
        color = (0, 255, 0)
        
        for obj in coordinates:
            xmin = int(obj[3] * width)
            ymin = int(obj[4] * height)
            xmax = int(obj[5] * width)
            ymax = int(obj[6] * height)
            
            updated_coords.append([xmin, ymin, xmax, ymax])
            
            # Draw rectangle box for the detected person
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=3)
        
        return image, updated_coords

    def preprocess_outputs(self, outputs):
        '''
        Pre-process the inference output to get the bounding box coordinates
        '''
        detected_coords = []
        
        for box in outputs[0][0]:
            confidence = box[2]
        
            # Get the box coordinates only if confidence is higher than the threshold
            if confidence >= self.threshold:
                detected_coords.append(box)
        
        return detected_coords

    def preprocess_input(self, image):
        '''
        Prepocess the input image - resize, transpose and reshape the image as needed
        '''
        dimention = (self.input_shape[3], self.input_shape[2])
        
        frame = cv2.resize(image, dimention)
        frame = frame.transpose((2,0,1))
        frame = frame.reshape(1, *frame.shape)
        
        return frame


def main(args):
    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()
    
    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            
            coords, image= pd.predict(frame)
            num_people= queue.check_coords(coords)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25
            
            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)
            
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)