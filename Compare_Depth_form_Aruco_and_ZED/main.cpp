#include "Aria.h"
#include <iostream>
#include <string>
#include <aruco/aruco.h>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

int main(int argc, char** argv){

  if (argc != 2){
    printf("Syntax is: %s <VGA or HD720 or HD1080 or HD2K>\n\n", argv[0]);
    return 1;
  }

  //*************************************************************************************************

  Aria::init(); // Initialize some global data
 
  ArArgumentParser parser(&argc, argv); // This object parses program options from the command line
  parser.loadDefaultArguments(); // Load some default values for command line arguments from /etc/Aria.args

  ArRobot robot; // Central object
  ArRobotConnector robotConnector(&parser, &robot); // Object that connects to the robot

  // Connect to the robot
  if (!robotConnector.connectRobot()){
    ArLog::log(ArLog::Terse, "-->Could not connect to the robot");
    if(parser.checkHelpAndWarnUnparsed()){
        Aria::logOptions();
        Aria::exit(1);
        return 1;
    }
  }
  if (!Aria::parseArgs()){
    Aria::logOptions();
    Aria::exit(1);
    return 1;
  }
     
  // Used to perform actions when keyboard keys are pressed
  ArKeyHandler keyHandler;
  Aria::setKeyHandler(&keyHandler);

  // Keydrive action
  ArActionKeydrive keydriveAct;
  keydriveAct.setSpeeds (200, 10);

  robot.attachKeyHandler(&keyHandler);
  robot.addAction(&keydriveAct, 100);

  robot.runAsync(true);
  std::cout << "\t\t-->Drive with Keyboard" << std::endl;
  robot.enableMotors();

  //*************************************************************************************************

  // Create a ZED camera object
  sl::Camera zed;

  // Set configuration parameters for the ZED
  sl::InitParameters init_params;
  if (std::string(argv[1]) == "VGA") 
    init_params.camera_resolution = sl::RESOLUTION::VGA;
  else if (std::string(argv[1]) == "HD720") 
    init_params.camera_resolution = sl::RESOLUTION::HD720;
  else if (std::string(argv[1]) == "HD1080") 
    init_params.camera_resolution = sl::RESOLUTION::HD1080;
  else if (std::string(argv[1]) == "HD2K") 
    init_params.camera_resolution = sl::RESOLUTION::HD2K;
  else 
    return 1;

  init_params.depth_mode = sl::DEPTH_MODE::QUALITY;
	init_params.camera_disable_self_calib = true;
  init_params.coordinate_units = sl::UNIT::METER;

  // Open the camera
  sl::ERROR_CODE err = zed.open(init_params);

  if (err != sl::ERROR_CODE::SUCCESS){
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

  //*************************************************************************************************

  aruco::CameraParameters CamParam;
  CamParam.CameraMatrix =  (cv::Mat_<float>(3,3) << fx_l, 0, cx_l, 0, fy_l, cy_l, 0, 0, 1);
  CamParam.Distorsion = (cv::Mat_<double>(1,5) << d0_l, d1_l, d2_l, d3_l, d4_l);

  // read marker size if specified (default value -1)
  float MarkerSize = 0.15;

  // Create the detector
  aruco::MarkerDetector MDetector;
  MDetector.setDictionary("ARUCO_MIP_36h12");
  MDetector.getParameters().detectEnclosedMarkers(true);

  cv::namedWindow("left", CV_WINDOW_NORMAL);
  cv::moveWindow("left", 0, 0);
  cv::resizeWindow("left", 672, 376);

  //*************************************************************************************************

  int iteration = 1;
  ArTime start;
  double tempo;

  float depth_value=0;

  std::ofstream distDect;
 
  for (;;){

    robot.lock();
    keydriveAct.activate();
    robot.unlock();
    ArUtil::sleep(100);

    if (keyHandler.getKey() == 97){
  
      std::cout << "\t\t-->Drive Automatic" << std::endl;
      keydriveAct.deactivate();
      ArUtil::sleep(100);
      distDect.open(std::string(argv[1])+".txt");

      while (iteration < 2){

        robot.lock();
        robot.setVel(iteration*100);
        robot.unlock();

        tempo = (3650/(iteration*100))*1000;

        start.setToNow();
        
        for(;;){
          
          robot.lock();

          if (start.mSecSince() > tempo){
            robot.unlock();
            cv::destroyWindow("left");
            distDect.close();
            break;
          }

          if (zed.grab() == sl::ERROR_CODE::SUCCESS){

            sl::Mat buffer_sl;
            cv::Mat buffer_cv;

            cv::Mat left_image;

            sl::Mat depth_map;
            
            // Get a new frame from camera
            zed.retrieveImage(buffer_sl, sl::VIEW::LEFT);
            // Retrieve depth
            zed.retrieveMeasure(depth_map, sl::MEASURE::DEPTH);

            buffer_cv = cv::Mat(buffer_sl.getHeight(), buffer_sl.getWidth(), CV_8UC4, buffer_sl.getPtr<sl::uchar1>(sl::MEM::CPU));
            buffer_cv.copyTo(left_image);

            cv::cvtColor(left_image, left_image, CV_RGBA2RGB);
                  
            // Ok, let's detect
            std::vector<aruco::Marker> Markers = MDetector.detect(left_image, CamParam, MarkerSize);

            // For each marker, draw info and its boundaries in the image
            for (unsigned int i = 0; i < Markers.size(); i++){
              // std::cout << Markers[i] << std::endl;
              Markers[i].draw(left_image, cv::Scalar(0, 0, 255), 2);
              // aruco::CvDrawingUtils::draw3dAxis(left_image, Markers[i], CamParam);
              // aruco::CvDrawingUtils::draw3dCube(left_image, Markers[i], CamParam);
            
              depth_map.getValue(Markers[i].getCenter().x,Markers[i].getCenter().y,&depth_value);
              // std::cout << "Marker= " << Markers[i].id << "  Center= (" << Markers[i].getCenter().x  << ", " << Markers[i].getCenter().y  <<")" << " Depth= "<< depth_value << std::endl;
              std::cout << depth_value << "\t" << Markers[i].Tvec.at<float>(0,2) << "\r" <<std::flush;
                if (distDect.is_open ())
                  distDect << depth_value << "\t" << Markers[i].Tvec.at<float>(0,2) << std::endl;
            }

            // show input with augmented information
            cv::imshow("left", left_image);
            // if(cv::waitKey(30) >= 0) break;
            cv::waitKey(1);
          }
          
          robot.unlock();
          ArUtil::sleep(1);
        } 

        robot.lock();
        robot.stop();
        robot.unlock();
        ArUtil::sleep(2000);

        //Telling the robot to move backwards one meter, then sleeping 5 seconds
        robot.lock();
        robot.move(-3550);
        robot.unlock();

        for(;;){
          robot.lock();
          if (robot.isMoveDone()){
            robot.unlock();
            break;
          }

          robot.unlock();
          ArUtil::sleep(1);
        }

        robot.lock();
        robot.stop();
        robot.unlock();
        ArUtil::sleep(2000);

        iteration++;
      }
      std::cout << "\n\t\t-->Drive with Keyboard" << std::endl;
      iteration = 1;
      robot.clearDirectMotion();
    }
  }
  
  zed.close();

  std::cout << "-->Shutting and exiting\n" << std::endl;
  Aria::exit(0);

  return 0;
}