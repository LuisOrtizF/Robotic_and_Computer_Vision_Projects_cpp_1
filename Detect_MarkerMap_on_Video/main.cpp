#include "aruco.h"
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <string>
#include <stdexcept>

int main(int argc, char** argv){
    
    try{
        if (argc < 4){
            std::cerr << "Usage: <video.avi> <markerMap.yml> <cameraParams.yml> <markerSize>" << std::endl;
            exit(0);
        }

        cv::VideoCapture vreader(argv[1]);
        if (!vreader.isOpened())
            throw std::runtime_error("Could not open input");

        // open video
        cv::Mat InImage, InImageCopy;

        // read marker map
        aruco::MarkerMap TheMarkerMapConfig;
        TheMarkerMapConfig.readFromFile(argv[2]);

        // read camera params
        aruco::CameraParameters CamParam;
        CamParam.readFromXMLFile(argv[3]);

        // read marker size
        float MarkerSize = -1;
        MarkerSize = static_cast<float>(atof(argv[4]));

        // Let go
        aruco::MarkerDetector MDetector;
        // set the appropiate dictionary type so that the detector knows it
        MDetector.setDictionary(TheMarkerMapConfig.getDictionary());

        aruco::MarkerMapPoseTracker MSPoseTracker;  // tracks the pose of the marker map
        // detect the 3d camera location wrt the markerset (if possible)
        if (TheMarkerMapConfig.isExpressedInMeters() && CamParam.isValid()){
            MSPoseTracker.setParams(CamParam, TheMarkerMapConfig);
        }

        cv::namedWindow("in", CV_WINDOW_NORMAL);
        cv::moveWindow("in", 0, 0);
        cv::resizeWindow("in", 672, 376);

        char key = 0;

        vreader >> InImage;

        do{
            // read input image(or first image from video)
            vreader.retrieve(InImage);
            InImage.copyTo(InImageCopy);

            // detect markers without computing R and T information
            std::vector<aruco::Marker> Markers = MDetector.detect(InImage);
            
            //CAMERA POSE wtr marker map
            if (MSPoseTracker.isValid())
                if (MSPoseTracker.estimatePose(Markers)){
                    //aruco::CvDrawingUtils::draw3dAxis(InImageCopy, CamParam, MSPoseTracker.getRvec(), MSPoseTracker.getTvec(), TheMarkerMapConfig[0].getMarkerSize()*2);
                    std::cout <<"Z-coordinate (cam-to-map) = "<< MSPoseTracker.getTvec().at<double>(2,0)<<std::endl;
                }

            // print the markers detected that belongs to the markerset
            std::vector<int> markers_from_set = TheMarkerMapConfig.getIndices(Markers); 

            for (auto idx : markers_from_set){
                Markers[idx].draw(InImageCopy, cv::Scalar(0, 0, 255), 2);
            }

            // show input with augmented information
            cv::imshow("in", InImageCopy);
            key = cv::waitKey(1);
            
        } while (key != 'q' && vreader.grab());

    }
    catch (std::exception& ex){
        std::cout << "Exception :" << ex.what() << std::endl;
    }
}