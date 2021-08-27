// c++ includes
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

// ZED includes
#include <sl/Camera.hpp>
#include "utils.hpp"

// Using namespace
using namespace sl;
using namespace std;

int main(int argc, char **argv)
{
    cout << "\nConvert_SVO(LeftCam)_Video_to_AVI\n" << endl;
 
    if (argc != 3 )
    {
        printf("Syntax is: %s <path_to_input_file.svo> <path_to_output_file.avi> \n", argv[0]);
        cout << "\nPress [Enter] to continue.\n" << endl;
        cin.ignore();
        return 1;
    }

    // Get input parameters
    string svo_input_path(argv[1]);
    string output_path(argv[2]);

    // Create ZED objects
    Camera zed;

    // Specify SVO path parameter
    InitParameters zedParams;
    zedParams.input.setFromSVOFile(svo_input_path.c_str());
    zedParams.depth_mode = DEPTH_MODE::QUALITY;
    zedParams.camera_disable_self_calib = true;
    zedParams.coordinate_units = UNIT::METER;
    
    // Open the SVO file specified as a parameter
    ERROR_CODE zed_open_state = zed.open(zedParams);

    if (zed_open_state != ERROR_CODE::SUCCESS)
    {
        std::cout << "\n\tCamera not open: " << zed_open_state << ". Exit program.\n" << std::endl;
        zed.close();
        return 1;
    }

    // Get image size
    Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
    int width = image_size.width;
    int height = image_size.height;

    sl::Mat left_image (width, height, MAT_TYPE::U8_C4, sl::MEM::CPU);
    cv::Mat left_image_cv_rgba = slMat2cvMat(left_image);
    cv::Mat left_image_cv_rgb(width, height, CV_8UC3);

    // Create video writer
    cv::VideoWriter* video_writer = 0;

    int fourcc = cv::VideoWriter::fourcc('M', '4', 'S', '2'); // MPEG-4 part 2 codec
    int frame_rate = fmax(zed.getInitParameters().camera_fps, 15); // Minimum write rate in OpenCV is 25
    video_writer = new cv::VideoWriter(output_path, fourcc, frame_rate, cv::Size(width, height));
    
    if (!video_writer->isOpened()) 
    {
        cout << "Error: OpenCV video writer cannot be opened. Please check the .avi file path and write permissions." << endl;
        zed.close();
        return 1;
    }

    RuntimeParameters rt_param;
    rt_param.sensing_mode = SENSING_MODE::STANDARD;

    // Start SVO conversion to AVI/SEQUENCE
    cout << "Converting SVO... Use Ctrl-C to interrupt conversion." << endl;

    int nb_frames = zed.getSVONumberOfFrames();
    cout<<"NumberOfFrames = "<<nb_frames<<endl;
    int svo_position = 0;

    SetCtrlHandler();

    zed.setSVOPosition(0);

    while (!exit_app){
        
        sl::ERROR_CODE err = zed.grab(rt_param);

        if (err == ERROR_CODE::SUCCESS){

            svo_position = zed.getSVOPosition();

            // Retrieve SVO images
            zed.retrieveImage(left_image, VIEW::LEFT, sl::MEM::CPU, image_size);

            // Convert SVO image from RGBA to RGB
            cv::cvtColor(left_image_cv_rgba, left_image_cv_rgb, cv::COLOR_RGBA2RGB);

            // Write the RGB image in the video
            video_writer->write(left_image_cv_rgb);

            // Display progress
            ProgressBar((float) (svo_position / (float) nb_frames), 30);
            
        }
        else if (err == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            cout << "\n\nSVO end has been reached. Exiting now.\n\n";
            exit_app = true;
        }
        else 
        {
            cout << "\n\nGrab Error: " << err << "\n" <<endl;
            exit_app = true;
        }
    }

    if (video_writer) 
    {
        // Close the video writer
        video_writer->release();
        delete video_writer;
    }

    zed.close();
    return 0;
}