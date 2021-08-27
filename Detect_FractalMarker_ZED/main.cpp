#include <iostream>
#include <string>
#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
#include <map>

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

  if (err != sl::SUCCESS){
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

  cv::namedWindow("left", CV_WINDOW_NORMAL);
  cv::moveWindow("left", 0, 0);
  cv::resizeWindow("left", 672, 376);

  sl::Mat buffer_sl;
  cv::Mat buffer_cv;
  cv::Mat left_image_rgba;
  cv::Mat left_image_rgb;

  //****************************************************************************************

  aruco::CameraParameters CamParam;
  CamParam.CameraMatrix =  (cv::Mat_<float>(3,3) << fx_l, 0, cx_l, 0, fy_l, cy_l, 0, 0, 1);
  CamParam.Distorsion = (cv::Mat_<double>(1,5) << d0_l, d1_l, d2_l, d3_l, d4_l);
  CamParam.CamSize = cv::Size(zed.getResolution().width, zed.getResolution().height);

  aruco::FractalDetector FDetector;
  FDetector.setConfiguration("FRACTAL_4L_6"); //0.187 FRACTAL_4L_6  0.185  FRACTAL_5L_6

  float MarkerSize = 0.1865;
  double distance = 0;

  if (CamParam.isValid())
    FDetector.setParams(CamParam, MarkerSize);

  cv::Mat tvec;

  for(;;){

      if (zed.grab() == sl::SUCCESS){
       
        // Get a new frame from camera
        zed.retrieveImage(buffer_sl, sl::VIEW_LEFT);
        buffer_cv = cv::Mat(buffer_sl.getHeight(), buffer_sl.getWidth(), CV_8UC4, buffer_sl.getPtr<sl::uchar1>(sl::MEM_CPU));
        buffer_cv.copyTo(left_image_rgba);

        cv::cvtColor(left_image_rgba, left_image_rgb, CV_RGBA2RGB);

        // Ok, let's detect
        if(FDetector.detect(left_image_rgb)){
          
          FDetector.drawMarkers(left_image_rgb);

          //get inner image points
          std::vector<aruco::Marker> Markers = FDetector.getMarkers();
          
          if(Markers.size() > 0){

            std::map<int, aruco::FractalMarker> id_fmarker = FDetector.getConfiguration().fractalMarkerCollection;
            std::vector<cv::Point2f> inners;
            std::map<int, std::vector<cv::Point3f>> id_innerCorners =  FDetector.getConfiguration().getInnerCorners();
            
            for(auto id_innerC:id_innerCorners){
                std::vector<cv::Point3f> inner3d;
                for(auto pt:id_innerC.second)
                    inners.push_back(cv::Point2f(pt.x,pt.y));
            }
            
            std::vector<cv::Point2f> srcPnts;
            std::vector<cv::Point2f> point2d;
            for(auto m:Markers)
            {
                for(auto p:id_fmarker[m.id].points)
                {
                    cv::Point3f p3d = p/( FDetector.getConfiguration().getFractalSize()/2);
                    srcPnts.push_back(cv::Point2f(p3d.x, p3d.y));
                }
                for(auto p:m)
                    point2d.push_back(p);
            }

            cv::Mat H;
            H = cv::findHomography(srcPnts, point2d);
            std::vector<cv::Point2f> dstPnt;
            cv::perspectiveTransform(inners, dstPnt, H);

            for(auto p:dstPnt)
              std::cout<<p<<std::endl;
          }
        }
          
        if(FDetector.poseEstimation()){
          FDetector.draw3d(left_image_rgb);
          tvec = FDetector.getTvec();
          printf("\nTranslation MARKER: Tx: %.3f, Ty: %.3f, Tz: %.3f\n", tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));

          // distance = sqrt(pow(tvec.at<double>(0,0),2) + pow(tvec.at<double>(1,0), 2) + pow(tvec.at<double>(2,0),2))*(double)MarkerSize/2;
          // std::cout << distance << std::endl;
        }
        else
          FDetector.draw2d(left_image_rgb); //Ok, show me at least the inner corners!
        
        // show input with augmented information
        cv::imshow("left", left_image_rgb);
        if(cv::waitKey(30) >= 0) break;
      }
  }

  zed.close();
  // Deinitialize camera in the VideoCapture destructor
  return 0;
}

// Puntos en U,V de los puntos externos de los amrcadores
// std::vector<aruco::Marker> Markers = FDetector.getMarkers();
// if(Markers.size() > 0)
// {
//   std::map<int, aruco::FractalMarker> id_fmarker = FDetector.getConfiguration().fractalMarkerCollection;
//   for(auto m:Markers)
//   {
//       for(auto p:m)
//           std::cout<<p<<std::endl;
//   }
// }
