#include <iostream>
#include <string>
#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>
#include <sl_zed/Camera.hpp>

int main(int argc, char** argv){

  if(argc != 2){
    std::cout << "Syntax is: %s <VGA or HD720 or HD1080 or HD2K>\n" << argv[0] << std::endl;
    return 1;
  }

  // Create a ZED camera object
  sl::Camera zed;

  // Set configuration parameters for the ZED
  sl::InitParameters init_params;
  if (std::string(argv[1]) == "VGA") 
    init_params.camera_resolution = sl::RESOLUTION_VGA;
  else if (std::string(argv[1]) == "HD720") 
    init_params.camera_resolution = sl::RESOLUTION_HD720;
  else if (std::string(argv[1]) == "HD1080") 
    init_params.camera_resolution = sl::RESOLUTION_HD1080;
  else if (std::string(argv[1]) == "HD2K") 
    init_params.camera_resolution = sl::RESOLUTION_HD2K;
  else 
    return 1;

  init_params.depth_mode = sl::DEPTH_MODE_QUALITY;
	init_params.camera_disable_self_calib = true;
  init_params.coordinate_units = sl::UNIT_METER;

  // Open the camera
  sl::ERROR_CODE err = zed.open(init_params);

  if (err != sl::SUCCESS) {
      printf("%s\n", sl::toString(err).c_str());
      zed.close();
      return 1; // Quit if an error occurred
  }

  sl::CalibrationParameters camInt = zed.getCameraInformation().calibration_parameters_raw;
  float fx_l = camInt.left_cam.fx;
  float fy_l = camInt.left_cam.fy;
  float cx_l = camInt.left_cam.cx;
  float cy_l = camInt.left_cam.cy;
  double d0_l = camInt.left_cam.disto[0];
  double d1_l = camInt.left_cam.disto[1];
  double d2_l = camInt.left_cam.disto[2];
  double d3_l = camInt.left_cam.disto[3];
  double d4_l = camInt.left_cam.disto[4];

  aruco::CameraParameters CamParam;
  CamParam.CameraMatrix =  (cv::Mat_<float>(3,3) << fx_l, 0, cx_l, 0, fy_l, cy_l, 0, 0, 1);
  CamParam.Distorsion = (cv::Mat_<double>(1,5) << d0_l, d1_l, d2_l, d3_l, d4_l);
  CamParam.CamSize = cv::Size(zed.getResolution().width, zed.getResolution().height);

  // read marker size if specified (default value -1)
  float MarkerSize = 0.15;

  // Create the detector
  aruco::MarkerDetector MDetector;
  MDetector.setDictionary("ARUCO_MIP_36h12");
  MDetector.setDetectionMode(aruco::DM_VIDEO_FAST);
  MDetector.getParameters().detectEnclosedMarkers(true);

  cv::namedWindow("left", CV_WINDOW_NORMAL);
  cv::moveWindow("left", 0, 0);
  cv::resizeWindow("left", 672, 376);

  for(;;) {

      if (zed.grab() == sl::SUCCESS) {

        sl::Mat buffer_sl;
        cv::Mat buffer_cv;

        cv::Mat left_image;
        
        // Get a new frame from camera
        zed.retrieveImage(buffer_sl, sl::VIEW_LEFT);

        buffer_cv = cv::Mat(buffer_sl.getHeight(), buffer_sl.getWidth(), CV_8UC4, buffer_sl.getPtr<sl::uchar1>(sl::MEM_CPU));
        buffer_cv.copyTo(left_image);

        cv::cvtColor(left_image, left_image, CV_RGBA2RGB);
              
        // Ok, let's detect
        std::vector<aruco::Marker> Markers = MDetector.detect(left_image, CamParam, MarkerSize);

        // for each marker, draw info and its boundaries in the image
        for (unsigned int i = 0; i < Markers.size(); i++)
        {
            std::cout << Markers[i] << std::endl;
            Markers[i].draw(left_image, cv::Scalar(0, 0, 255), 2);
        }

        // draw a 3d cube in each marker if there is 3d info
        if (CamParam.isValid() && MarkerSize != -1)
          for (unsigned int i = 0; i < Markers.size(); i++)
          {
            aruco::CvDrawingUtils::draw3dAxis(left_image, Markers[i], CamParam);
            aruco::CvDrawingUtils::draw3dAxis(left_image, Markers[i], CamParam);
          }

        // show input with augmented information
        cv::imshow("left", left_image);
        if(cv::waitKey(30) >= 0) break;
      }
  }

  zed.close();
  // Deinitialize camera in the VideoCapture destructor
  return 0;
}
