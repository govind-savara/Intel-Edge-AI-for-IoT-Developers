# Suggestions to Make Your Project Stand Out!

1. Can you improve your inference speed without significant drop in performance by changing the precision of some of the models? In your README, include a short write-up that explains the procedure and the experiments you ran to find out the best combination of precision.

2. Write test cases and ensure your code covers all the edge cases.

3. Benchmark the running times of different parts of the preprocessing and inference pipeline and let the user specify a CLI argument if they want to see the benchmark timing. Use the `get_perf_counts` API to print the time it takes for each layer in the model.

4. Use Async Inference to allow multiple inference pipelines. Show how this affects performance and power as a short writeup in the README file.

5. There will be certain edge cases that will cause your system to not function properly. Examples of this include: lighting changes, multiple people in the same input frame, and so on. Make changes in your preprocessing and inference pipeline to solve some of these issues. Write a short write-up in the README about the problem it caused and your solution.

6. Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the camera feed back on). Show how this affects performance and power as a short write up in the README file.

7. Build an inference pipeline for both video file and webcam feed as input. Allow the user to select their input option in the command line arguments.