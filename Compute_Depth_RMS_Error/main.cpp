#include <iostream>
#include <sstream>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <aruco/aruco.h>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>

#include <ucoslam/ucoslam.h>
#include <ucoslam/mapviewer.h>
#include <ucoslam/stereorectify.h>

cv::Mat slMat2cvMat(sl::Mat &input) 
{
    int cv_type = -1;

    switch (input.getDataType()) 
    {
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

    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU));
}

std::vector <cv::Affine3d> getPoses(std::string filename)
{
    std::vector <cv::Affine3d> T_gt;
    std::ifstream poses;
    std::string file;
    poses.open(filename);

    if (poses.is_open())
    {
        cv::Affine3d T;
            
        while (getline(poses, file))
        {
            std::istringstream file_stream (file);
            double item_stream;
            std::vector<double> file_vector; 

            while(file_stream>>item_stream)
                file_vector.push_back(item_stream);

            cv::Vec3d t;
            t << file_vector[1], file_vector[2], file_vector[3];
            T.translation(t);

            //std::cout << file_vector[1] << " " <<file_vector[2] << " " <<file_vector[3] << " " << std::endl;

            Eigen::Quaterniond q;

            q.x() = file_vector[4];
            q.y() = file_vector[5];
            q.z() = file_vector[6];
            q.w() = file_vector[7];

            Eigen::Matrix3d Q = q.toRotationMatrix();
            cv::Matx33d R33;

            for (int v = 0; v < 3; v++) 
                for (int u = 0; u < 3; u++) 
                    R33(u,v) = Q(u,v);

            T.rotation(R33);
            
            //std::cout << T.matrix << std::endl;
            T_gt.push_back(T);
        }
        poses.close(); 
    }
    else
        std::cout << "\n\tUnable to open file!" << std::endl;
    return T_gt;
}

std::vector <uint64_t> getTimestamps(std::string filename)
{
    std::vector <uint64_t> timestamps;
    std::ifstream poses;
    std::string file;
    poses.open(filename);

    if (poses.is_open())
    {            
        while (getline(poses, file))
        {
            std::istringstream file_stream (file);
            double item_stream;
            std::vector<double> file_vector; 

            while(file_stream>>item_stream)
                file_vector.push_back(item_stream);

            uint64_t timestamp = file_vector[0];
            //std::cout << file_vector[0] << std::endl;

            timestamps.push_back(timestamp);
        }

        poses.close(); 
    }
    else
        std::cout << "\n\tUnable to open file!" << std::endl;

    return timestamps;
}

cv::Point3f computeMarkerCenter(std::vector<cv::Point3f> points3D)
{
    //Compute 3D TRANSLATION of the MARKER CENTER w.t.r. THE MM ORIGIN 
    float base = ((points3D[1].x - points3D[0].x)/2.0);
    float height = ((points3D[3].y - points3D[0].y)/2.0);

    float x_center = points3D[0].x + base;
    float y_center = points3D[0].y + height;

    float sum_Z = 0;

    for (size_t i = 0; i < points3D.size(); i++)
        sum_Z += points3D[i].z;
    
    float z_center = sum_Z / points3D.size();

    cv::Point3f center;
    center.x = x_center;
    center.y = y_center;
    center.z = z_center;

    return center;
}

int main(int argc, char **argv) 
{
    if (argc != 6 )
    {
        printf("\n\n\tSyntax is: %s <video.svo> <optimizedMarkersMap.yml> <arucoConfig.yml> <gtPoses.txt> <zedPoses.txt>", argv[0]);
        std::cout << "\n\n\tPress [Enter] to continue.\n" << std::endl;
        std::cin.ignore();
        return 1;
    }

    // ZED
    sl::InitParameters zedParams;
    zedParams.input.setFromSVOFile(argv[1]);
    zedParams.depth_mode = sl::DEPTH_MODE::QUALITY;
    zedParams.camera_disable_self_calib = true;
    zedParams.coordinate_units = sl::UNIT::METER;

    // Open the ZED
    sl::Camera zed;
    sl::ERROR_CODE zed_open_state = zed.open(zedParams);
    if (zed_open_state != sl::ERROR_CODE::SUCCESS) 
    {
        std::cout << "\n\n\tCamera Open: " << zed_open_state << ". Exit program.\n" << std::endl;
        zed.close();
        return 1;
    }

    auto resolution = zed.getCameraInformation().camera_configuration.resolution;
    cv::Size size(resolution.width, resolution.height);

    zed.setSVOPosition(1);

    sl::CalibrationParameters zedCamParams = zed.getCameraInformation().calibration_parameters;
    float fx_l = zedCamParams.left_cam.fx;
    float fy_l = zedCamParams.left_cam.fy;
    float cx_l = zedCamParams.left_cam.cx;
    float cy_l = zedCamParams.left_cam.cy;
    double d0_l = zedCamParams.left_cam.disto[0];
    double d1_l = zedCamParams.left_cam.disto[1];
    double d2_l = zedCamParams.left_cam.disto[2];
    double d3_l = zedCamParams.left_cam.disto[3];
    double d4_l = zedCamParams.left_cam.disto[4];

    // ARUCO
    // read camera parameters
    aruco::CameraParameters arucoCamParams;
    arucoCamParams.CameraMatrix = (cv::Mat_<float>(3,3) << fx_l, 0, cx_l, 0, fy_l, cy_l, 0, 0, 1);
    arucoCamParams.Distorsion = (cv::Mat_<float>(1,5) << d0_l, d1_l, d2_l, d3_l, d4_l);
    arucoCamParams.CamSize = size;
    // read marker maps
    aruco::MarkerMap optimizedMap;
    optimizedMap.readFromFile(argv[2]);
    // tracks the pose of the marker map
    aruco::MarkerMapPoseTracker poseTracker;
    float markerSize = cv::norm(optimizedMap[0][0] - optimizedMap[0][1]);
    // detect the 3d camera location wrt the markerset (if possible)
    if (optimizedMap.isExpressedInMeters() && arucoCamParams.isValid())
        poseTracker.setParams(arucoCamParams, optimizedMap, markerSize);
    std::vector<int> markersIDs;
    optimizedMap.getIdList(markersIDs, true);
    aruco::MarkerDetector arcuoDetector;
    arcuoDetector.loadParamsFromFile(argv[3]);

    // OpenCV
    int window_h = 600;
    int window_w = 800;
    sl::Mat image_sl_l (size.width, size.height, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);
    cv::Mat image_cv_rgba_l = slMat2cvMat(image_sl_l);

    cv::namedWindow("Left", CV_WINDOW_NORMAL);
	cv::moveWindow("Left", 0, 0);
	cv::resizeWindow("Left", window_w, window_h-200);

    // Create 3D window
    cv::viz::Viz3d visualizer("Visual Odometry");
	visualizer.setBackgroundColor(cv::viz::Color::white());
    visualizer.setWindowPosition(cv::Point(0, window_h+45));
    visualizer.setWindowSize(cv::Size(window_w, window_h));
    visualizer.showWidget("origin", cv::viz::WCoordinateSystem(0.5));
    cv::viz::WText3D text = cv::viz::WText3D("G(0,0,0)", cv::Point3d( 0.0,  0.3, 0.0), 0.1, true, cv::viz::Color::black());
    visualizer.showWidget("text", text);

    for(auto idx : markersIDs)
    {
        std::vector<cv::Point3f> points3D = optimizedMap.getMarker3DInfo(idx).points;
        for(int j = 0; j <  4; j++)
        {
            if(j==3)
            {
                cv::viz::WLine line = cv::viz::WLine(points3D[j], points3D[0], cv::viz::Color::green());
                visualizer.showWidget("linesOur" + std::to_string(idx) + std::to_string(j), line);
                cv::viz::WText3D markerID = cv::viz::WText3D(std::to_string(idx), points3D[j], 0.1, true, cv::viz::Color::black());
                visualizer.showWidget("markerID_"+std::to_string(idx), markerID);
            }
            else
            {
                cv::viz::WLine line = cv::viz::WLine(points3D[j], points3D[j+1], cv::viz::Color::green());	
                visualizer.showWidget("linesOur" + std::to_string(idx) + std::to_string(j), line);
            }
            visualizer.spinOnce(1, true);
        }
    }
    
    // Load GT CAMERA POSES
    std::vector<cv::Affine3d> T_gt;
    T_gt = getPoses((std::string)argv[4]);
    std::vector<uint64_t> gtTimestamps;
    gtTimestamps = getTimestamps((std::string)argv[4]);

    // Load ZED CAMERA POSES
    std::vector<cv::Affine3d> T_zed;
    T_zed = getPoses((std::string)argv[5]);
    std::vector<uint64_t> zedTimestamps;
    zedTimestamps = getTimestamps((std::string)argv[5]);

    // SAVE RESULTS
    std::string videoDir (argv[1]);
    std::string outputDir = videoDir.substr(0, videoDir.find_last_of("/")+1) + "output/"; 
    //std::cout<<outputDir<<std::endl;
    std::string videoName = videoDir.substr(videoDir.find_last_of("/")+1);
    videoName = videoName.substr(0, videoName.find_last_of("."));
    //std::cout<<videoName<<std::endl;

    if(!boost::filesystem::is_directory(outputDir))
        boost::filesystem::create_directory(outputDir);

    std::ofstream r3D_data;
    r3D_data.open(outputDir + videoName + "_r3D_data.txt", std::ofstream::out | std::ofstream::trunc);

    // VARIABLES
    char key = ' ';
    sl::Mat zedCloud;
    int index=0;

    // Start SVO playback
    while (key != 'q' && !visualizer.wasStopped()) 
    {
        sl::ERROR_CODE zed_error = zed.grab();

		if (zed_error == sl::ERROR_CODE::SUCCESS) 
        {      
            zed.retrieveImage(image_sl_l, sl::VIEW::LEFT, sl::MEM::CPU, sl::Resolution(size.width, size.height));
            cv::Mat image_cv_rgb_l;
            cv::cvtColor(image_cv_rgba_l, image_cv_rgb_l, CV_RGBA2RGB);
            cv::Mat image_cv_rgb_l_out;
            image_cv_rgb_l.copyTo(image_cv_rgb_l_out);
            // Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieveMeasure(zedCloud, sl::MEASURE::XYZRGBA);
            uint64_t timestamp = zedTimestamps[index];

            //std::cout<<timestamp<<std::endl;
            std::vector<uint64_t>::iterator it = std::find(gtTimestamps.begin(), gtTimestamps.end(), timestamp);
            // int ind = std::distance(gtTimestamps.begin(), it);            
            // if ( ind >= gtTimestamps.size())
            //     continue;
            // std::cout << gtTimestamps[ind] << std::endl;

            if (it != gtTimestamps.end())
            {
                std::cout << "Timestamp Found" << std::endl;
                // Detection of the markers
                std::vector<aruco::Marker> detectedMarkers = arcuoDetector.detect(image_cv_rgb_l, arucoCamParams, markerSize);
                
                if(detectedMarkers.size()>0)
                {
                    float sum_error_x=0,sum_error_y=0,sum_error_z=0;
                    int samples=1;

                    for (auto idx : optimizedMap.getIndices(detectedMarkers))
                    {
                        // detectedMarkers[idx].draw(image_cv_rgb_l_out, cv::Scalar(0, 0, 255), 2);
                        // aruco::CvDrawingUtils::draw3dCube(image_cv_rgb_l_out, detectedMarkers[idx], arucoCamParams);
                        // aruco::CvDrawingUtils::draw3dAxis(image_cimage_cv_rgb_l_outv_rgb_l, detectedMarkers[idx], arucoCamParams);
                        // // std::cout << detectedMarkers[idx].id << " ";

                        // Compute GT 2D control points -> 4 marker's corners and the center
                        std::vector < cv::Point2f > gtPoints2D;
                        // Compute GT 3D control points -> 4 marker's corners and the center
                        std::vector<cv::Point3f> contour3D = optimizedMap.getMarker3DInfo(detectedMarkers[idx].id).points;

                        std::vector < cv::Point3f > gtPoints3D;

                        for(int i = 0; i < detectedMarkers[idx].size(); i++)
                        {
                            gtPoints2D.push_back(detectedMarkers[idx][i]);
                            gtPoints3D.push_back(contour3D[i]);
                        }

                        gtPoints2D.push_back(detectedMarkers[idx].getCenter());
                        cv::Point3f center3D = computeMarkerCenter(contour3D);
                        gtPoints3D.push_back(center3D);

                        // Visualization GT 3D points
                        cv::viz::WCloud gtPoints3DCloud = cv::viz::WCloud(gtPoints3D, cv::viz::Color::green());
                        gtPoints3DCloud.setRenderingProperty( cv::viz::POINT_SIZE, 5);
                        std::string gtCloud_id = "gt3d_"+std::to_string(index)+"_"+std::to_string(detectedMarkers[idx].id);
                        visualizer.showWidget(gtCloud_id, gtPoints3DCloud);
                        visualizer.spinOnce(1, true);
                        visualizer.removeWidget(gtCloud_id);

                        // Get ZED 3D control points -> 4 marker's corners and the center i.e. get the 3D point cloud values for pixel (i,j)
                        std::vector < cv::Point3f > zedPoints3D;

                        int subsamples=0;

                        for(int i = 0; i < gtPoints2D.size(); i++)
                        {
                            //Visualization GT 2D points
                            circle( image_cv_rgb_l_out, gtPoints2D[i], 5, cv::Scalar(0,255,0),5,8);
                            
                            size_t x = gtPoints2D[i].x;
                            size_t y = gtPoints2D[i].y;                        
                            sl::float4 p3D_zed;
                            zedCloud.getValue(x,y,&p3D_zed);
                            cv::Point3f point3D_zed;
                            point3D_zed.x = p3D_zed.x;
                            point3D_zed.y = p3D_zed.y;
                            point3D_zed.z = p3D_zed.z;

                            if(std::isfinite(point3D_zed.z))
                            {                      
                                cv::Point3f point3D = T_zed[index]*point3D_zed;
                                float error_x = std::pow((gtPoints3D[i].x - point3D.x),2);
                                float error_y = std::pow((gtPoints3D[i].y - point3D.y),2);
                                float error_z = std::pow((gtPoints3D[i].z - point3D.z),2);
                                sum_error_x += error_x;
                                sum_error_y += error_y;
                                sum_error_z += error_z;
                                samples+=subsamples;
                                subsamples++;
                            }
                        }

                        // Visualization ZED 3D points
                        if (zedPoints3D.size() > 0)
                        {
                            cv::viz::Color point3DZedColor = cv::Scalar(173, 216, 230); //lightblue
                            cv::viz::WCloud zedPoints3DCloud = cv::viz::WCloud(zedPoints3D, point3DZedColor);
                            zedPoints3DCloud.setRenderingProperty( cv::viz::POINT_SIZE, 5);
                            std::string zedCloud_id = "zed3d_"+std::to_string(index)+"_"+std::to_string(detectedMarkers[idx].id);
                            visualizer.showWidget(zedCloud_id, zedPoints3DCloud);
                            //visualizer.removeWidget(zedCloud_id);
                        }
                    }
                    if (samples > 1)
                    {
                        float rmse_x = std::sqrt(sum_error_x/samples);  
                        float rmse_y = std::sqrt(sum_error_y/samples);
                        float rmse_z = std::sqrt(sum_error_z/samples);
                        //SAVE DATA
                        //timestamp-traslation x-traslation y-traslation z-rmse_x-rmse_y-rmse_z
                        //The points are referenced to G(0,0,0)
                        r3D_data << timestamp << " " << T_gt[index].translation().val[0] << " " << T_gt[index].translation().val[1]  << " " << T_gt[index].translation().val[2] << " " 
                        << rmse_x << " " << rmse_y << " " << rmse_z << "\n";
                    }
                }
            }
            else
                std::cout << "Timestamp Not Found" << std::endl;

            cv::String tex = "Timestamp: " + std::to_string(timestamp);
            cv::putText(image_cv_rgb_l_out, tex, cv::Point2d(20,resolution.height-20), CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,255,0), 5);
            cv::imshow("Left", image_cv_rgb_l_out);
            key = cv::waitKey(1);
            visualizer.spinOnce(1, true);
            if(index == zedTimestamps.size())
                break;
            index++;

        }
        else if (zed_error == sl::ERROR_CODE::END_OF_SVOFILE_REACHED)
        {
            std::cout << "\n\n\tSVO end has been reached.\n" << std::endl;
            break;
        }
        else 
        {
            std::cout << "\n\n\tGrab ZED : " << zed_error << ".\n" << std::endl;
            break;
        }
    }

    zed.close();
    r3D_data.close();

    while(key != 'q' && !visualizer.wasStopped())
    {
        key = cv::waitKey(1);
        visualizer.spinOnce(1, true); 
    }
        
    return 0;
}