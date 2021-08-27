# Robotic and Computer Vision Projects  with C++
---

This repository contains projects of computer vision tasks implemented with __*C++*__ libraries.

<div align="center">
<table> 
  <tr>
    <td> <img src="/Images/opencv_logo.png" width="150" height="125" alt="opencv_logo" /> </td>
    <td> <img src="/Images/pcl_logo.png" width="150" height="150" alt="pcl_logo" /> </td>
    <td> <img src="/Images/cmake_logo.png" width="200" height="80" alt="cmake_logo" /></td>
  </tr>
</table>
</div>

<div align="center">
<table> 
<tr> 
<td> 

+ **Capture_Data_Kinect1_Kinect2 (Registered)_ZED**: Using the __*Kinect 1*__, __*Kinect 2*__ or a __*Stereolabs __*ZED*__*__ camera, you can save undistorted RGB images (*.png*), depth maps (*.png*) and point clouds (*.pcd*). Additionally, save the camera calibration parameters (*.txt*). The RGB images of the __*Kinect 2*__ are registered (RGB-depth) to their depth map.
  + **how use**: `./capture __*ZED*__`
  + **output**: folder `/build/SavedFiles/` with all capture data.
 
<tr>
<td> 

+ **Capture_Data_Kinect1_Kinect2 (UnRegistered)_ZED**: Using the __*Kinect 1*__, __*Kinect 2*__ or a __*Stereolabs __*ZED*__*__ camera, you can save undistorted RGB images (*.png*), depth maps (*.png*) and point clouds (*.pcd*). Additionally, save the camera calibration parameters (*.txt*). The RGB images of the __*Kinect 2*__ are unregistered (RGB-depth) to their depth map.
  + **how use**: `./capture __*ZED*__`
  + **output**: folder `/build/SavedFiles/` with all capture data.

<div align="center">
<table> 
  <tr> 
    <td> <img src="/Images/zed.jpg" width="150" height="125" alt="zed" /> </td>
    <td> <img src="/Images/kinect_v1.jpg" width="150" height="50" alt="kinect_v1" /> </td>
    <td> <img src="/Images/kinect_v2.jpeg" width="150" height="125" alt="kinect_v2" /></td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Chessboards_Corners_Detection**: Detect and plot the corners of a chessboard using __*OpenCV*__.
  + **how use**: `./corners`

<tr>
<td> 

+ **Compare_Depth_form_Aruco_and_ZED**: Comparison between the depth measured using an __*Aruco*__ marker and the depth measured with the __*Stereolabs __*ZED*__*__ Camera API. The camera is mounted in a robot __*Pioneer 3AT*__. The robot can be drive in two modes: manual (with keyboard) and automatic.
  + **how use**: `./compareDepth` *`VGA`* or *`HD720`* or *`HD1080`* or *`HD2K`*

<tr>
<td> 

+ **Compute_3D_Normals_Keypoints_Correspondences**: Compute the normals, keypoints and correspondences in point clouds using __*PCL*__.
  + **how use**: `./correspondences3D robot1.pcd` *`<normals`* or *`keypoints`* or *`correspondences>`*

<div align="center">
<table> 
  <tr> 
    <td> <img src="/Images/normals3D.png" width="600" height="100" alt="normals3D" /> </td>
    <td> <img src="/Images/keypoints3D.png" width="600" height="100" alt="keypoints3D" /> </td>
    <td> <img src="/Images/correspondences3D.png" width="600" height="100" alt="correspondences3D" /> </td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Compute_Depth_Error_in_RGBD_Sensors**: Detect and plot the corners of a chessboard in real-time using __*OpenCV*__ and a webcam.
  + **inputs**:
    + __*device*__: `Kv1` or `Kv2` or `ZED_WVGA`
    + __*save_viewers*__: `off` or `on`
    + __*flag*__: `after` or `before`
  + **how use**: `./depth_error ../Data/ <device> <save_viewers> <flag>`

<tr>
<td> 

+ **Compute_Depth_RMS_Error**: Computes the depth error between the ground truth (__*Smart Markers*__) and the depth computed by a __*ZED*__ camera.
  + **inputs**:
    + __*e1*__: video in (*.svo*) format, recorded with a __*ZED*__ camera
    + __*mm_our*__: optimized markers map get by __*Smart Markers*__
    + __*arucoConfig_7*__: __*Aruco*__ detection configuration file
    + __*e1_gt_poses*__: ground truth camera poses 
    + __*e1_zed_poses*__: camera poses get by __*ZED*__ API
  + **output**:
    + __*output*__: file `/input/output/e1_r3D_data.txt` with timestamp and camera poses (3D position and quaternion).
  + **how use**: `./depth_rmse ../input/e1.svo ../input/mm_our.yml ../input/arucoConfig_7.yml ../input/e1_gt_poses.txt ../input/e1_zed_poses.txt`

<div align="center">
<table> 
  <tr>
    <td> <img src="/Images/depth_rmse.gif" width="150" height="125" alt="depth_rmse" /> </td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Compute_Essential_and_Fundamental_Matrix**: Compute the Essential and Fundamental matrix using __*OpenCV*__.
  + **how use**: `./essential_fundamental` 

<div align="center">
<table> 
  <tr> 
    <td> <img src="/Images/fundamental.png" width="600" height="100" alt="fundamental" /> </td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Compute_Hausdorff_Distance**: Compute Hausdorff distance between two point clouds.
  + **how use**: `./hausdorff ../Data/input_cloud.pcd ../Data/taget_cloud.pcd`

+ **Convert_SVO(LeftCam)_Video_to_AVI**: 
  + **inputs**:
    + __*svo_video*__: video in (*.svo*) format, recorded with a __*ZED*__ camera
    + __*avi_video*__: name for the output (*.avi*) video
  + **how use**: `./svo2avi <svo_video> <output_video>`
  + **example**: `./svo2avi ../Data/test.svo ../Data/test.avi`

<tr>
<td> 

+ **Correct_Sitting_Position**: Demonstrates using __*PCL*__ and __*ICP*__ algorithm to correct sitting position using point clouds.
  + **inputs**: 
    + __*initial_cloud*__: point cloud of a person in correct sitting position
    + __*target_cloud*__: point cloud of a person in correct sitting position
    + __*mode_visualization*__: `0` bi-color or `1` rgb-color
  + **how use**: `./correct_position initial_cloud.pcd target_cloud 1`

<div align="center">
<table> 
  <tr> 
    <td> <img src="/Images/sitting_correction.jpg" width="600" height="200" alt="sitting_correction" /> </td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Crop_Images**: Crop images using __*OpenCV*__.
  + **input**: folder `../Images` with the images to be crop
  + **output**: folder `../CroopedImages` with the crooped images
  + **how use**: `./crop ../Images/`

<tr>
<td> 

+ **Detect_Aruco_Marker**: Simple __*Aruco*__ marker detection on a image.
  + **how use**: `./aruco_image ../Images/test_0.jpg`

<tr>
<td> 

+ **Detect_Aruco_with_UVC_Cam**: Simple __*Aruco*__ marker detection in real-time using a webcam.
  + **how use**: `./aruco_webcam`

<tr>
<td> 

+ **Detect_Aruco_with_ZED**: Simple __*Aruco*__ marker detection in real-time using __*ZED*__ camera.
  + **how use**: `./aruco_zed` *`VGA`* or *`HD720`* or *`HD1080`* or *`HD2K`*

<tr>
<td> 

+ **Detect_FractalMarker_ZED**: Simple __*Aruco*__ fractal marker detection in real-time using __*ZED*__ camera.
  + **how use**: `./fractal` *`VGA`* or *`HD720`* or *`HD1080`* or *`HD2K`*

<tr>
<td> 

+ **Detect_MarkerMap_on_Video**: Track a camera using Aruco Marker Map detection.
  + **inputs**:
    + __*mm.avi*__: video with Aruco markers
    + __*mono_params.yml*__: intrinsic parameters of a monocular camera
    + __*marker_size*__: markers size in meters
  + **how use**: `./mm_track ../Data/mm.avi ../Data/mm_our.yml ../Data/mono_params.yml 0.1295`

<div align="center">
<table> 
  <tr>
    <td> <img src="/Images/mm_track.gif" width="150" height="125" alt="mm_track" /> </td>
  </tr>
</table>
</div>

<tr>
<td> 

+ **Extract_3D_Plane_from_ZED_Video**: Extract and visualize the most representative 3D plane in a point cloud.
  + **how use**: `./extract_plane ../Data/test_0.svo`

<tr>
<td> 

+ **Extract_3D_Planes_from_PCL_Cloud**: Segment the largest 3D planar components from a point cloud

  + **how use**: `./extract_planes`
  + **outputs**:
    + __*../Data/downsampled_cloud.pcd*__: point cloud after filtering
    + __*../Data/detected_plane_0.pcd*__: point cloud representing the planar component

</table>
</div>

## Installation:

+ Dependences (mandatory):
    + __*ZED*__ SDK 3.0
    + __*CUDA*__ 10.0
    + __*OpenCV*__ 3.4.1
    + __*Aruco*__ 3.0.12
    + __*PCL*__ 1.8
    + __*ARIA*__ or __*ARIACODA*__
    + __*UcoSLAM*__ 
    + __*Boost*__
    + __*Freenect2*__
    + __*Libviso2*__
    + __*Open-NI*__
    + __*Libusb*__ 1.0

+ Download any project and open a terminal _`ctrl+t`_:
    ```
    $ cd path 
    $ mkdir build & cd build 
    $ cmake .. 
    $ make
    ```

## NOTE:

| If you find any of these codes helpful, please share my __[GitHub](https://github.com/LuisOrtizF)__ and __*STAR*__ :star: this repository to help other enthusiasts to find these tools. Remember, the knowledge must be shared. Otherwise, it is useless and lost in time.|
| :----------- |