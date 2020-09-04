[![Intel® Edge AI for IoT Developers](../images/Udacity-Intel_Edge_AI_for_IoT_Developers_logo.svg)](https://www.udacity.com/course/intel-edge-ai-for-iot-developers-nanodegree--nd131)

# Computer Pointer Controller

![Mouse Pointer Controller](./images/mouse_pointer_controller.gif)

In this project used the Gaze Detection model to control the mouse pointer of computer. 
I used [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model to estimate gaze of the user's eye and changed the mouse pointer position accordingly.
This application uses Inference Engine API of Intel's OpenVINO Toolkit.  
The Gaze estimation model requires three inputs:
- The head pose
- The left eye image
- The right eye image

To get these inputs I used to the following three models.
1. [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
3. [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

This model is tested on  
**Operating System**: Windows 10 Pro  
**Processor**: Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz 2.90 GHz  
**GPU**: Intel(R) HD Graphics 620  Driver Version: 22.20.16.4771

# Project Pipeline
The below images shows the pipeline of data flow

![Project Pipeline](./images/project_pipeline.png)

## Project Set Up and Installation
- Download and install the **OpenVINO Toolkit**. The installation direction of OpenVINO can be found [here](https://docs.openvinotoolkit.org/latest/index.html)
- Create a virtual environment for this application  
`py -m venv intel-edge`
- Activate virtual environment  
`.\intel-edge\Scripts\activate`
- Install all the required python packages which are listed in 'requirement.txt' file.  
`pip install -r requirements.txt`
- Download the required models
  - [Face Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)  
  `python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name face-detection-adas-binary-0001`
  
  - [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)  
  `python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name head-pose-estimation-adas-0001`
  
  - [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)  
  `python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name landmarks-regression-retail-0009`
  
  - [Gaze Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)  
  `python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name gaze-estimation-adas-0002`
  
- Project structure  `tree /F`
```
.
│   .Instructions.md.swp
│   README.md
│   requirements.txt
│
├───bin
│       .gitkeep
│       demo.mp4
│
├───models
│   └───intel
│       ├───face-detection-adas-binary-0001
│       │   └───FP32-INT1
│       ├───gaze-estimation-adas-0002
│       │   ├───FP16
│       │   ├───FP16-INT8
│       │   └───FP32
│       ├───head-pose-estimation-adas-0001
│       │   ├───FP16
│       │   ├───FP16-INT8
│       │   └───FP32
│       └───landmarks-regression-retail-0009
│           ├───FP16
│           ├───FP16-INT8
│           └───FP32
│
└───src
        base_model.py
		face_detection.py
        facial_landmarks_detection.py
        gaze_estimation.py
        head_pose_estimation.py
        input_feeder.py
        main.py
        model.py
        mouse_controller.py
```

## Demo
To run the basic demo of the model use the following commands.

- Activate the created virtual environment  
`.\intel-edge\Scripts\activate`

- Set the OpenVINO environment variables by executing 'setupvars.bat' file.  
`"C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"`

- To view all the installed python packages we can use the execute the below command  
`pip freeze`

- Traverse to the project folder  
`cd computer_pointer_controller`

- Run the Application using the following command  
  - To run on CPU  
  ```
  python src\main.py 
  --fd_model models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 
  --hp_model models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001 
  --fl_model models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 
  --ge_model models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 
  --input bin\demo.mp4 
  --prob_threshold 0.6 
  --mouse_precision high 
  --mouse_speed fast 
  --display
  ```
  - To run on GPU  
  ```
  python src\main.py 
  --fd_model models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 
  --hp_model models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001 
  --fl_model models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 
  --ge_model models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 
  --input bin\demo.mp4 
  --prob_threshold 0.6 
  --mouse_precision high 
  --mouse_speed fast 
  --device GPU 
  --display
  ```
  - To use WEBCAM  
  ```
  python src\main.py 
  --fd_model models\intel\face-detection-adas-binary-0001\FP32-INT1\face-detection-adas-binary-0001 
  --hp_model models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001 
  --fl_model models\intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009 
  --ge_model models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 
  --input CAM 
  --prob_threshold 0.6 
  --mouse_precision high 
  --mouse_speed fast 
  --device GPU 
  --display
  ```

## Documentation
```
usage: main.py [-h] -fd FD_MODEL -hp HP_MODEL -fl FL_MODEL -ge GE_MODEL -i
               INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD]
               [-mp MOUSE_PRECISION] [-ms MOUSE_SPEED] [-dis] [-np]

optional arguments:
  -h, --help            show this help message and exit
  -fd FD_MODEL, --fd_model FD_MODEL
                        Path to a trained Face Detection model.
  -hp HP_MODEL, --hp_model HP_MODEL
                        Path to a trained Head Post Estimation model.
  -fl FL_MODEL, --fl_model FL_MODEL
                        Path to a trained Facial Landmarks Detection model.
  -ge GE_MODEL, --ge_model GE_MODEL
                        Path to a trained Gaze Estimation model.
  -i INPUT, --input INPUT
                        Path to image/video file or 'CAM' for webcam input
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plug-in for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering (0.5 by
                        default)
  -mp MOUSE_PRECISION, --mouse_precision MOUSE_PRECISION
                        Set the precision for mouse movement: high, low,
                        medium.
  -ms MOUSE_SPEED, --mouse_speed MOUSE_SPEED
                        Set the speed for mouse movement: fast, slow, medium.
  -dis, --display       Flag to display intermediate model outputs.
  -np, --no_pointer_move
                        Flag for not to move the mouse pointer for the gaze
                        estimation.
```

## Benchmarks

|                       | CPU      |   GPU    |
|-----------------------|----------|----------|
| Total Load time       | 00:00:00 | 00:00:25 |
| Total Inference time  | 00:01:22 | 00:01:21 |


## Results
- The model load time for GPU is more than CPU.
- The load time for model with FP32 precision is less than FP16. Also FP16 precision loading time is less than INT8. 
- The inference time for model with FP32 is less than the FP16 and INT8 precisions.
- Due to the low precision the accuracy of INT8 and FP16 is less than FP32.

## Stand Out Suggestions
The following some points are stand out suggestions that we can attempt.
- Improving inference speed without significant drop in performance.
- Use the VTune Amplifier to find hotspots in our inference engine pipeline.
- Building an inference engine for both video file and webcam stream as inputs.

## Edge Cases
The model is able to use human gaze to control the computer pointer. There are some edge cases with this model, when there
are multiple persons detected then the model will consider the highest threshold face detected by [Face Detection Model](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html). 
Also in case the pointer reached the extreme edge of the monitor then the model may breakdown.
