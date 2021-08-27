#include <iostream>
#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){

  // Open the ZED camera
  cv::VideoCapture cap(1);
  if(!cap.isOpened()){
      std::cout <<"Could not open input" << std::endl;
      return -1;
  }

  // Set the video resolution
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 1344);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 376);

  aruco::CameraParameters CamParam;
  // read camera parameters if specifed
  CamParam.readFromXMLFile("../Data/CamParam.yml");

  // read marker size if specified (default value -1)
  float MarkerSize = 0.15;

  // Create the detector
  aruco::MarkerDetector MDetector;
  MDetector.setDictionary("ARUCO_MIP_36h12");
  MDetector.getParameters().detectEnclosedMarkers(true);

  cv::namedWindow("left", CV_WINDOW_NORMAL);
  cv::moveWindow("left", 0, 0);
  cv::resizeWindow("RGB", 672, 376);

  for(;;)
  {
      cv::Mat frame, left_image;
      // Get a new frame from camera
      cap >> frame;
      // Extract left and right images from side-by-side
      left_image = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
      
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
        }

      // show input with augmented information
      cv::imshow("left", left_image);
      if(cv::waitKey(30) >= 0) break;
  }
  // Deinitialize camera in the VideoCapture destructor
  return 0;
}
