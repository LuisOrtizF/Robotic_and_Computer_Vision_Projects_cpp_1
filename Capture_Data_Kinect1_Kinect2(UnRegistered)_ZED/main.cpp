#include <iostream>

#include <boost/filesystem.hpp>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

//kv2
#include "libfreenect2/libfreenect2.hpp"
#include "libfreenect2/frame_listener_impl.h"
#include "libfreenect2/registration.h"
#include "libfreenect2/packet_pipeline.h"
#include <libfreenect2/logger.h>

//ZED includes
#include <sl/Camera.hpp>

using namespace cv;
using namespace std;
using namespace boost::filesystem;
using namespace sl;

string pathRGB, pathDepth;

int num_frames = 30; // Number of frames to capture
int count_save = 1;
char key = ' ';
int new_width, new_height;
double minVal, maxVal;

ofstream file_calib_param;

string GetLinuxBuildPath(){
    char buf[PATH_MAX + 1];
    if(readlink("/proc/self/exe", buf, sizeof(buf) - 1) == -1)
        throw string("readlink() failed");
    string str(buf);
    return str.substr(0, str.rfind('/'));
}

// Conversion function between sl::Mat and cv::Mat
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}

bool CaptureKv1()
{
    VideoCapture sensor1;
    sensor1.open(CAP_OPENNI);

    if(!sensor1.isOpened())
    {
        cout << "\nCan't open device.\n" << endl;
        return false;
    }

    sensor1.set(CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ);  
    sensor1.set(CAP_PROP_OPENNI_REGISTRATION, false);
    
    new_width  = sensor1.get(CAP_PROP_FRAME_WIDTH);
    new_height = sensor1.get(CAP_PROP_FRAME_HEIGHT);

    namedWindow("RGB", WINDOW_NORMAL);
    moveWindow("RGB", 0, 0);
    resizeWindow("RGB", new_width, new_height);

    namedWindow("Depth", WINDOW_NORMAL);
    moveWindow("Depth", new_width + 85, 0 );
    resizeWindow("Depth", new_width, new_height);

    cv::Mat depth1, bgr1; //cloud1;

    create_directory(pathRGB);
    create_directory(pathDepth);

    if (file_calib_param.is_open ())
        file_calib_param    << 522.2590 << " " << 0.0 << " " << 330.1798 << " " << 0.0 << " "
                            << 0.0 << " " << 523.4194 << " " << 254.4374 << " " << 0.0 << " "
                            << 0.0 << " " <<  0.0 << " " <<  1.0 << " " << 0.0 << endl;


    cout << "Capture with Kinect v1." << endl;


    while (key != 'q') 
    {
        if(sensor1.grab())
        {
            sensor1.retrieve(bgr1, CAP_OPENNI_BGR_IMAGE); //Retrieve Color Image CV_8UC3 RGB
            imshow("RGB", bgr1);

            sensor1.retrieve(depth1, CAP_OPENNI_DEPTH_MAP); //Depth in mm CV_16UC1
            
            //Only for visualization
            minMaxLoc(depth1, &minVal, &maxVal); //find minimum and maximum intensities
            depth1.convertTo(depth1, CV_8UC1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
            imshow("Depth", depth1);
                
            if (key == 's' || key == 'S')
            {
                create_directory(pathRGB + to_string(count_save));
                create_directory(pathDepth + to_string(count_save));
                //create_directory(pathCloud + to_string(count_save));
        
                for(int i = 1; i <= num_frames; i++)
                {
                    if(sensor1.grab())
                    {      
                        sensor1.retrieve(bgr1, CAP_OPENNI_BGR_IMAGE);
                        imwrite(pathRGB + to_string(count_save)+ "/" + to_string(i) + string(".png"),bgr1); 
                        
                        sensor1.retrieve(depth1, CAP_OPENNI_DEPTH_MAP);   
                        imwrite(pathDepth + to_string(count_save)+ "/" + to_string(i) + string(".png"),depth1);

                    }
                    else
                    {  
                        cout << "Device can't grab images." << endl;
                        return false;
                    }
                }
                count_save++;
            }
        }
        else
        {  
            cout << "Device can't grab images." << endl;
            return false;
        }
        key = waitKey(10);
    }
    sensor1.release();
}

bool CaptureKv2()
{
    libfreenect2::Freenect2 sensor2;
    libfreenect2::Freenect2Device *dev = 0;

    libfreenect2::setGlobalLogger(NULL); //Hide devide infromation on terminal screen

    if(sensor2.enumerateDevices() == 0){
      cout << "No device connected." << endl;
      return false;
    }
    
    string serial = sensor2.getDefaultDeviceSerialNumber();
    
    dev = sensor2.openDevice(serial);

    if(dev == 0){
      cout << "Failure opening device." << endl;
      return false;
    }

    int types = 0;
    types |=  libfreenect2::Frame::Color | libfreenect2::Frame::Depth;

    libfreenect2::SyncMultiFrameListener listener(types);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    dev->start();

    new_width = 512;
    new_height = 424;

    namedWindow("RGB", WINDOW_NORMAL);
    moveWindow("RGB", 0, 0);
    resizeWindow("RGB", new_width, new_height);

    namedWindow("Depth", WINDOW_NORMAL);
    moveWindow("Depth", new_width + 85, 0 );
    resizeWindow("Depth", new_width, new_height);
   
    create_directory(pathRGB);
    create_directory(pathDepth);

    libfreenect2::Freenect2Device::IrCameraParams depthParams = dev->getIrCameraParams();

    if (file_calib_param.is_open ())
        file_calib_param    << depthParams.fx << " " << 0.0 << " " << depthParams.cx << " " << 0.0 << " "
                            << 0.0 << " " << depthParams.fy << " " << depthParams.cy << " " << 0.0 << " "
                            << 0.0 << " " <<  0.0 << " " <<  1.0 << " " << 0.0 << endl;


    cout << "Capture with Kinect v2." << endl;

    cv::Mat bgrmat, depthmat;

    while(key != 'q')
    {
        listener.waitForNewFrame(frames);

        libfreenect2::Frame *bgr = frames[libfreenect2::Frame::Color]; //Color 1920x1080. BGRX or RGBX.
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth]; //512x424 float, unit: millimeter. Non-positive, NaN, and infinity are invalid or missing data.
        
        cv::Mat(bgr->height, bgr->width, CV_8UC4, bgr->data).copyTo(bgrmat);        
        flip(bgrmat, bgrmat, 1);
        cvtColor(bgrmat, bgrmat , COLOR_BGRA2BGR);
        imshow("RGB", bgrmat);

        cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);
        //Only for depth visualization
        flip(depthmat, depthmat, 1);    
        minMaxLoc(depthmat, &minVal, &maxVal); //find minimum and maximum intensities
        depthmat.convertTo(depthmat, CV_8UC1, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
        imshow("Depth", depthmat);  

        listener.release(frames);

        if (key == 's' || key == 's'){

            create_directory(pathRGB + to_string(count_save));
            create_directory(pathDepth + to_string(count_save));

            for(int i = 1; i <= num_frames; i++)
            {
                    listener.waitForNewFrame(frames);

                    libfreenect2::Frame *bgr = frames[libfreenect2::Frame::Color];
                    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

                    //Save RGB image on .png file
                    cv::Mat(bgr->height, bgr->width, CV_8UC4, bgr->data).copyTo(bgrmat);        
                    flip(bgrmat, bgrmat, 1);
                    cvtColor(bgrmat, bgrmat, COLOR_BGRA2BGR);
                    imwrite(pathRGB + to_string(count_save)+ "/" + to_string(i) + string(".png"), bgrmat);   

                    // Save depth map on .png file 
                    cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);
                    depthmat.convertTo(depthmat, CV_16UC1);
                    flip(depthmat, depthmat, 1);
                    imwrite(pathDepth + to_string(count_save) + "/" + to_string(i) + string(".png"), depthmat); 

                    listener.release(frames);                                                   
            }
            count_save++;
        }
    
        key = waitKey(10);
    }

    dev->stop();
    dev->close();
}

bool CaptureZED(int zed_resolution)
{    
    // Create a ZED camera object
    Camera sensor3;
    
    InitParameters init_params;

    switch(zed_resolution)
    {
        case 3:
            init_params.camera_resolution = RESOLUTION::VGA;
            break;
        case 4:
            init_params.camera_resolution = RESOLUTION::HD720;
            break;
        case 5:
            init_params.camera_resolution = RESOLUTION::HD1080;
            break;
        case 6:
            init_params.camera_resolution = RESOLUTION::HD2K;
            break;
    }

    init_params.depth_mode = DEPTH_MODE::QUALITY;
    init_params.camera_disable_self_calib = true; // Take the optional calibration parameters without optimizing them
    init_params.coordinate_units = UNIT::MILLIMETER;

    // Open the camera
    ERROR_CODE err = sensor3.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        cout << toString(err) << endl;
        sensor3.close();
        return false; //Quit if an error occurred
    }

    // Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

    sl::CameraInformation cam_info = sensor3.getCameraInformation();
    sl::Resolution zed_res = cam_info.camera_configuration.resolution;

    new_width = zed_res.width;
    new_height = zed_res.height;

    sl::CalibrationParameters calib_param = cam_info.camera_configuration.calibration_parameters;

    float fx_l = calib_param.left_cam.fx;
    float fy_l = calib_param.left_cam.fy;
    float cx_l = calib_param.left_cam.cx;
    float cy_l = calib_param.left_cam.cy;

    if (file_calib_param.is_open ())
        file_calib_param    << fx_l << " " << 0.0 << " " << cx_l << " " << 0.0 << " "
                            << 0.0 << " " << fy_l << " " << cy_l << " " << 0.0 << " "
                            << 0.0 << " " <<  0.0 << " " <<  1.0 << " " << 0.0 << endl;

    sl::Mat image_left_zed(new_width, new_height, sl::MAT_TYPE::U8_C4);
    sl::Mat depth_image_zed(new_width, new_height, sl::MAT_TYPE::U8_C4);

    namedWindow("RGB", WINDOW_NORMAL);
    moveWindow("RGB", 0, 0);
    resizeWindow("RGB", new_width/3, new_height/3);

    namedWindow("Depth", WINDOW_NORMAL);
    moveWindow("Depth", new_width/3  + 400, 0 );
    resizeWindow("Depth", new_width/3, new_height/3);

    cout << "Capture with ZED in all resolutions." << endl;

    while (key != 'q') 
    {
        if (sensor3.grab(runtime_parameters) == ERROR_CODE::SUCCESS) 
        {
            // Retrieve the left image, depth image
            sensor3.retrieveImage(image_left_zed, VIEW::LEFT, MEM::CPU, zed_res);
            sensor3.retrieveImage(depth_image_zed, VIEW::DEPTH, MEM::CPU, zed_res);

            cv::Mat image_left_ocv = slMat2cvMat(image_left_zed);
            imshow("RGB", image_left_ocv);
            cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
            imshow("Depth", depth_image_ocv);

            if (key == 's' || key == 'S')
            {
                create_directory(pathRGB + to_string(count_save));
                create_directory(pathDepth + to_string(count_save));
                //create_directory(pathCloud + to_string(count_save));

                for(int i = 1; i <= num_frames; i++)
                {
                    if (sensor3.grab(runtime_parameters) == ERROR_CODE::SUCCESS) 
                    {
                        cv::Mat left_image (new_width, new_height, CV_8UC4);
                        sl::Mat buffer_sl;
                        cv::Mat buffer_cv;
                        sensor3.retrieveImage(buffer_sl, VIEW::LEFT);
                        buffer_cv = cv::Mat(buffer_sl.getHeight(), buffer_sl.getWidth(), CV_8UC4, buffer_sl.getPtr<sl::uchar1>(sl::MEM::CPU));
                        buffer_cv.copyTo(left_image);
                        cvtColor(left_image, left_image, COLOR_RGBA2RGB);
                        imwrite(pathRGB + to_string(count_save)+ "/" + to_string(i) + string(".png"), left_image);                                                                          
                        sl::Mat depth_map;
                        sensor3.retrieveMeasure(depth_map, MEASURE::DEPTH);
                        depth_map.write((pathDepth + to_string(count_save) + "/" +to_string(i)).c_str());               
                    }
                }
                count_save++;
            }      
        }

        key = waitKey(10);
    }

    sensor3.close();
    destroyAllWindows();
    key = '\0';
}

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        printf("\nSyntax is: %s <Kv1 or Kv2 or ZED> \n\n", argv[0]);
        return 1;
    }

    string buildPath = GetLinuxBuildPath().c_str();
    string savedPath =  buildPath + "/SavedFiles/";
    create_directory(savedPath);

    string sensor [] = {"Kv1", "Kv2", "ZED", "ZED_WVGA", "ZED_720", "ZED_1080", "ZED_2K"};

    file_calib_param.open(savedPath + "calib_params.txt");

    if (string(argv[1]) == sensor[0]) 
    {
        pathRGB = savedPath + "rgb_" + sensor[0]+ "/";
        pathDepth = savedPath + "depth_" + sensor[0]+ "/";
        if(CaptureKv1());       
    }
    else if (string(argv[1]) == sensor[1])
    {
        pathRGB = savedPath + "rgb_" + sensor[1]+ "/";
        pathDepth = savedPath + "depth_" + sensor[1]+ "/";
        if(CaptureKv2());
    }
    else if (string(argv[1]) == sensor[2])
    {   
        for (int i = 3; i < 7; i++)
        {
            pathRGB = savedPath + "rgb_" + sensor[i]+ "/";
            pathDepth = savedPath + "depth_" + sensor[i]+ "/";

            create_directory(pathRGB);
            create_directory(pathDepth);

            //Count the number of subfolders in one directory
            int dir_count = 0;
            directory_iterator end_iter;

            for (directory_iterator dir_itr(pathRGB); dir_itr != end_iter; dir_itr++)
                if (is_directory(dir_itr->status()))
                    dir_count++;

            count_save = dir_count+1;
            
            if(CaptureZED(i));
        }
        
        
    }
    else cerr << "\nUnsupported devide.\n" << endl; 

    file_calib_param.close();

    return 0;
}