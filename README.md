# carbot_inference

The real-time inference engine for **carbot**, my self-driving RC car. This is what runs
the trained model on the Jetson, fast enough to actually drive.

It pulls frames straight off the camera, preprocesses them on the GPU, runs the model
through TensorRT, and turns the output into driving waypoints. Built on DeepStream /
GStreamer + CUDA so it keeps up in real time. The ROS2 side (carbot_ws) wraps this to
actually steer the car.

**Stack:** C++ · CUDA · TensorRT · DeepStream · GStreamer · Jetson Orin Nano

## Part of the carbot project

- [carbot_drivetrain](https://github.com/tomludbrook10/carbot_drivetrain) — ESP32 drivetrain firmware
- [carbot_ws](https://github.com/tomludbrook10/carbot_ws) — the ROS2 brain that runs the live driving loop
- [carbot_action_model](https://github.com/tomludbrook10/carbot_action_model) — trains the image→action model
- [carbot_teleoperation](https://github.com/tomludbrook10/carbot_teleoperation) — remote driving + data recording
