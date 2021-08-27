//standard includes
#include <iostream>

//Opencv includes
#include <opencv2/opencv.hpp>

//PCL includes
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/transformation_estimation_svd.h>

//libcbdetect
#include "rt_nonfinite.h"
#include "dCP.h"
#include "dCP_emxAPI.h"

//Manage files paths
#include <boost/filesystem.hpp>

//ZED includes
//#include <sl_zed/Camera.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;
//using namespace sl;

//Chessboards Sizes
Size chess_size;

//Internal corners
int num_corners_chess = 28;

//Chess board Square size(mm)
const float side_h = 149, side_w = 148;

//Chessboard borders
const float border_h = 149, border_w = 56;

//Corners Coordinates in Image System (u,v) pixels
vector<Point2f> corners_uv;

//Clouds Viewer
boost::shared_ptr<pcl::visualization::PCLVisualizer> viz_clouds;

//Results File
ofstream DepthRMSE;
ofstream DepthStdDevE;
ofstream Distances;

// Conversion function between cv::Mat and sl::Mat

// sl::Mat cvMat2slMat(cv::Mat input) {
//     // Mapping between MAT_TYPE and sl_type
//     sl::MAT_TYPE sl_type;
//     switch (input.type()) {
//         case CV_32FC1 : sl_type = MAT_TYPE_32F_C1; break;
//         case CV_32FC2 : sl_type = MAT_TYPE_32F_C2; break;
//         case CV_32FC3 : sl_type = MAT_TYPE_32F_C3; break;
//         case CV_32FC4 : sl_type = MAT_TYPE_32F_C4; break;
//         case CV_8UC1 : sl_type = MAT_TYPE_8U_C1; break;
//         case CV_8UC2 : sl_type = MAT_TYPE_8U_C2; break;
//         case CV_8UC3 : sl_type = MAT_TYPE_8U_C3; break;
//         case CV_8UC4 : sl_type = MAT_TYPE_8U_C4; break;
//         default: break;
//     }

//     return sl::Mat(input.cols, input.rows, sl_type, MEM_CPU);
// }

//Find string into string array
static bool in_array(const string &value, const vector<string> &array)
{
    return find(array.begin(), array.end(), value) != array.end();
}

static bool FindCorners(cv::Mat src_aux, cv::Mat bgrmat)
{
    emxArray_real_T *corners;
    emxArray_uint8_T *image;
    double tam_chess[2];
    emxInitArray_real_T(&corners, 2); //(filas,columnas)

    int w = src_aux.cols, h = src_aux.rows;
    static int iv2 [2] = {h, w};
    image = emxCreateND_uint8_T(2, *(int(*)[2])&iv2[0]);

    int idx0, idx1;

    for(idx0 = 0; idx0 < h; idx0++) 
        for(idx1 = 0; idx1 < w; idx1++) 
            image->data[idx0 + h * idx1] = src_aux.at<uchar>(idx0, idx1);

    dCP(image, corners, tam_chess);

    if(corners->size[0] == num_corners_chess)
    {
        for(idx0 = 0; idx0 < corners->size[0]; idx0++)
            corners_uv.push_back(Point2f(corners->data[idx0], corners->data[idx0 + corners->size[0]]));

        chess_size = Size(tam_chess[1]-1, tam_chess[0]-1);

        drawChessboardCorners(bgrmat, chess_size, cv::Mat(corners_uv), true);
        
        //cout << "Find: " << corners->size[0] << " Corners." << endl;
        
        return true;
    }

    else return false;

    emxDestroyArray_real_T(corners);
    emxDestroyArray_uint8_T(image);
}

static void CreateIdealCLoud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ideal_cloud)
{
    //CREATE IDEAL CLOUD WITH COODINATES OF THE CORNERS IN THE WORLD-OBJECT SYSTEM (mm)
    //number of internal corners of the plane (x-y) 3D chessboard (x(cols-width), y(rows-height))
    int x, y, i;

    //Corners Coordinates in Camera System (X,Y,Z) mm
    vector<Point3f> corners_world;

    //7 width x 4 height 
    for (x = chess_size.width - 1; x >=0; x--)
        for (y = 0; y < chess_size.height; y++)
            corners_world.push_back(Point3f(border_w + x * side_w, border_h + y * side_h, 0));

    // Create ideal point cloud
    uint8_t r(0), g(0), b(0);

    for (i = 0; i < corners_world.size(); i++)
    {
        pcl::PointXYZRGB point;
        point.x = corners_world[i].x;
        point.y = corners_world[i].y;
        point.z = corners_world[i].z;
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        point.rgb = *reinterpret_cast<float*>(&rgb);
        ideal_cloud->points.push_back (point);
        //cout << point.x << " " << point.y << " " <<point.z <<endl;
    }

    ideal_cloud->width = (int) ideal_cloud->points.size ();
    ideal_cloud->height = 1;
}

static void CreateRealCloud(vector<string> intrinsics, cv::Mat depthmatUnd, string data_dir_depth_full, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &real_cloud)
{
    // FIND REAL COORDINATES OF THE CORNERS IN THE CAMERA SYSTEM (mm)
    // CONVERT PIXEL COORDINATES TO CAMERA COORDINATES
    // Create real point cloud

    Point3f point_3d;

    float X,Y,Z = 0;

    uint8_t r(255), g(0), b(0);

    // depthmatUnd.convertTo(depthmatUnd, CV_32FC1);
    // sl::uchar1 *ptrcv = depthmatUnd.data;
    // size_t 	stepcv = depthmatUnd.step;
    // sl::Mat depth_image_zed (depthmatUnd.cols, depthmatUnd.rows, MAT_TYPE_32F_C1, ptrcv, stepcv, MEM_CPU);
   
   float sum_Z = 0;

    for (int i = 0; i < corners_uv.size(); i++)
    {
        Z = depthmatUnd.at<unsigned short>(cvRound(corners_uv[i].y), cvRound(corners_uv[i].x));
        //depth_image_zed.getValue(cvRound(corners_uv[i].x), cvRound(corners_uv[i].y), &Z);
        X = ((corners_uv[i].x - strtof(intrinsics[2].c_str(),0)) / strtof(intrinsics[0].c_str(),0)) * Z;
        Y = ((corners_uv[i].y - strtof(intrinsics[6].c_str(),0)) / strtof(intrinsics[5].c_str(),0)) * Z;
        
        //cout << cvRound(corners_uv[i].y) << " " << cvRound(corners_uv[i].x) << " " << Z << endl;

        pcl::PointXYZRGB point;
        point.x = X;
        point.y = Y;
        point.z = Z;
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        point.rgb = *reinterpret_cast<float*>(&rgb);
        real_cloud->points.push_back (point);

        sum_Z = sum_Z + Z;
    }

    Distances << sum_Z/corners_uv.size() << endl; 

    real_cloud->width = (int) real_cloud->points.size ();
    real_cloud->height = 1;
}

static void Clouds_Register(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ideal_cloud, 
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &real_cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &final_cloud)
{
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> ASVD;
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB,pcl::PointXYZRGB>::Matrix4 transform_matrix;
    ASVD.estimateRigidTransformation (*real_cloud, *ideal_cloud, transform_matrix);
    pcl::transformPointCloud(*real_cloud, *final_cloud, transform_matrix);
}

static void Depth_RMSE(  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ideal_cloud,
                        pcl::PointCloud<pcl::PointXYZRGB>::Ptr &final_cloud)
{
    double dist, dist_x, dist_y, dist_z, valx, valy, valz;
    double sum_total = 0;

    for(size_t i = 0; i < final_cloud->points.size(); i++)
    {
        // if(final_cloud->points[i].z > 500 && final_cloud->points[i].z < 12000)
        valx = final_cloud->points[i].x - ideal_cloud->points[i].x;
        valy = final_cloud->points[i].y - ideal_cloud->points[i].y;
        valz = final_cloud->points[i].z - ideal_cloud->points[i].z;

        dist_x = pow(valx,2);
        dist_y = pow(valy,2);
        dist_z = pow(valz,2);

        dist = dist_x + dist_y + dist_z;

        sum_total = sum_total + dist;
    }

    //RMSE
    double RMSE =  pow(sum_total/final_cloud->points.size(), 0.5);
    cout <<"RMSE: " << RMSE << endl;

    DepthRMSE << RMSE << endl;
}

static void Depth_StdDevE( pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ideal_cloud,
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr &final_cloud )
{

    double dist, dist_x, dist_y, dist_z, valx, valy, valz, mean_distance;
    double sum_total = 0;

    for(size_t i = 0; i < final_cloud->points.size(); i++)
    {
        valx = final_cloud->points[i].x - ideal_cloud->points[i].x;
        valy = final_cloud->points[i].y - ideal_cloud->points[i].y;
        valz = final_cloud->points[i].z - ideal_cloud->points[i].z;

        dist_x = pow(valx, 2);
        dist_y = pow(valy, 2);
        dist_z = pow(valz, 2);

        dist = dist_x + dist_y + dist_z;
        dist = pow(dist, 0.5);

        sum_total = sum_total + dist;
    }

    mean_distance = sum_total/final_cloud->points.size();

    sum_total = 0;

    for(size_t i = 0; i < final_cloud->points.size(); i++)
    {
        valx = final_cloud->points[i].x - ideal_cloud->points[i].x;
        valy = final_cloud->points[i].y - ideal_cloud->points[i].y;
        valz = final_cloud->points[i].z - ideal_cloud->points[i].z;

        dist_x = pow(valx,2);
        dist_y = pow(valy,2);
        dist_z = pow(valz,2);

        dist = dist_x + dist_y + dist_z;
        dist = pow(dist, 0.5);

        sum_total = sum_total + pow(dist - mean_distance, 2);
    }

    //StdDev
    double StdDevE =  pow(sum_total/(final_cloud->points.size()-1), 0.5);
    cout <<"StdDevE: " << StdDevE << endl;

    DepthStdDevE << StdDevE << endl;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> init_visualizator ()
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer());
    
    // Initial frame
    viewer->addCoordinateSystem (1000);
    viewer->setBackgroundColor (0.75, 0.75, 0.75);

    viewer->setShowFPS (false);	
    viewer->setSize (640,480);

    viewer->setWindowName("Ideal and Real Clouds Register");
    viewer->setPosition(800,0);

    //pos_x, pos_y, pos_z, view_x, view_y, view_z, up_x, up_y, up_z	
    viewer->setCameraPosition(0.0,0.0,5000,0.0,0.0,0.0,0.0,0.0,0.0);

    //Add axes labels
    pcl::PointXYZRGB point;
    point.getArray3fMap () << 1100, 0, 0;
    viewer->addText3D ("Xw", point, 60, 1, 0, 0, "x_");
    point.getArray3fMap () << 0, 1100, 0;
    viewer->addText3D ("Yw", point, 60, 0, 1, 0, "y_");
    point.getArray3fMap () << 0, 0, 1200;
    viewer->addText3D ("Zw", point, 60, 0, 0, 1, "z_");

    return (viewer);
}

static void ViewSave(string especific_results_path_full,
int j, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &ideal_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &final_cloud,
cv::Mat bgrmat)
{
    //Visualization of Corners and write image
    imwrite(especific_results_path_full + "corners_" + to_string(j) + ".png", bgrmat);
    imshow("Corners Detection", bgrmat);
    waitKey(2000);

    viz_clouds->removeAllPointClouds();
    viz_clouds->addPointCloud<pcl::PointXYZRGB>(ideal_cloud, "ideal_cloud");
    viz_clouds->addPointCloud<pcl::PointXYZRGB>(final_cloud, "final_cloud");
    viz_clouds->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,"ideal_cloud");
    viz_clouds->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5,"final_cloud");
    viz_clouds->saveScreenshot(especific_results_path_full + "register_" + to_string(j) + ".png");
    viz_clouds->spinOnce(1);
}

int main(int argc, char* argv[])
{
    cout<<"\nCompute Depth Error of RGB-D Sensors\n"<<endl;

    if(argc != 5)
    {
        cout    << "Usage:'" << string(argv[0]) << " <data_path> <device> <save_viewers> <marker>'\n"
                << "        ->device: 'Kv1'\n"
                << "                : 'Kv2'\n"
                << "                : 'ZED_WVGA'\n"
                << "        ->save_viewers: 'off'\n"
                << "                      : 'on'\n"
                << "        ->flag: 'after'\n"
                << "                : 'before'\n"
                << endl;
        return 1;
    }

    //Load data directory
    string data_dir = string(argv[1]);
    path data_path(data_dir);
    
    if (!exists(data_path))
    {
        cerr << "Not found data directory: " << data_path << endl;
        return 1;
    }
    
    string data_dir_rgb =  string(argv[1]) + "rgb_" + string(argv[2]);
    string data_dir_depth =  string(argv[1]) + "depth_" + string(argv[2]);

    path data_path_rgb(data_dir_rgb);
    path data_path_depth(data_dir_depth);

    if (!exists(data_path_rgb) && !exists(data_path_depth))
    {
        cerr << "Not found data rgb or depth in: " << data_dir_rgb << " or " << data_dir_depth << endl;
        return 1;
    }

    //Verify device
    vector<string> device {"Kv1", "Kv2", "ZED_WVGA"};

    for (int i = 0; i < device.size(); i++)
    {
        if (!in_array(string(argv[2]), device)) 
        {
            cerr << "Unsupported devide.\n" << endl; 
            return 1;
        }
    }

    //Read devices intrinsic parameters
    if (!exists(data_dir + "/calib_params.txt"))
    {
        cerr << "Could not read file: 'calib_params.txt'" << endl;
        return 1;
    }

    ifstream file_calib_param;
    file_calib_param.open(data_dir + "/calib_params.txt");
    string line;
    vector<string> intrinsics;

    for(int lineno = 0; getline (file_calib_param,line) && lineno < device.size(); lineno++)
        for (int i = 0; i < device.size(); i++)
            if(lineno == i && device[i] == string(argv[2]))
                boost::split(intrinsics, line, boost::is_any_of("\t "));

    file_calib_param.close(); 

    cv::Mat bgrmat_gray;
    bool find_corners = false;
    dCP_initialize();

    //Create ideal cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr corners_ground_truth (new pcl::PointCloud<pcl::PointXYZRGB>);
    //Create real cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr corners_real (new pcl::PointCloud<pcl::PointXYZRGB>);
    //Final cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr corners_align (new pcl::PointCloud<pcl::PointXYZRGB>);

    //Create visualization windows
    namedWindow("Corners Detection", WINDOW_NORMAL);
    moveWindow("Corners Detection", 0, 0);
    resizeWindow("Corners Detection", 640, 480);
    viz_clouds = init_visualizator();

    //Create save directories and files
    string results_path, especific_results_path_full, especific_results_path;
    results_path = current_path().string() + "/Results/";
    create_directory(results_path);

    if (string(argv[4]) == "after")
        especific_results_path = results_path + string(argv[2]) + "_after/";
    if (string(argv[4]) == "before")
        especific_results_path = results_path + string(argv[2]) + "_before/";
    
    create_directory(especific_results_path);

    //Load calibartion parameters
    DepthRMSE.open(especific_results_path + "RMSE_" + string(argv[2]) + ".txt");
    DepthStdDevE.open(especific_results_path + "StdDevE_" + string(argv[2]) + ".txt");
    Distances.open(especific_results_path + "Distances_" + string(argv[2]) + ".txt");
    
    //Count the number of subfolders in one directory
    string data_dir_rgb_full, data_dir_depth_full;
    int dir_count = 0;
    directory_iterator end_iter;

    //Color and Depth map
    cv::Mat bgrmat, depthmatUnd;

    for (directory_iterator dir_itr(data_dir_rgb); dir_itr != end_iter; dir_itr++)
        if (is_directory(dir_itr->status()))
            dir_count++; //dir_count+1

    for(int i = 1; i < dir_count+1; i++) //Distance
    {
        for(int j = 1; j < 3; j++) //Repetibilidade
        {
            //Read RGB image and its correspondent depth map
            //_COLOR - always convert image to the 3 channel BGR color image.
            //_ANYDEPTH - return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.

            data_dir_rgb_full = data_dir_rgb + "/" + to_string(i) + "/" + to_string(j) + string(".png");                                          
            bgrmat = imread (data_dir_rgb_full, IMREAD_COLOR); // read and convert image in 3 channel BGR.        
            cvtColor(bgrmat, bgrmat_gray, COLOR_BGR2GRAY);           
            
            data_dir_depth_full = data_dir_depth + "/" + to_string(i) + "/" + to_string(j) + string(".png");                                                    
            depthmatUnd = imread (data_dir_depth_full, IMREAD_ANYDEPTH);
            
            find_corners = FindCorners(bgrmat_gray, bgrmat);

            if(find_corners)
            {
                CreateIdealCLoud(corners_ground_truth);
                CreateRealCloud(intrinsics, depthmatUnd, data_dir_depth_full, corners_real);
                Clouds_Register(corners_ground_truth, corners_real, corners_align);
                Depth_RMSE(corners_ground_truth, corners_align);
                Depth_StdDevE(corners_ground_truth, corners_align);

                if(string(argv[3]) == "on")
                {
                    especific_results_path_full = especific_results_path + to_string(i) +"/";
                    create_directory(especific_results_path_full);
                    ViewSave(especific_results_path_full, j, corners_ground_truth, corners_align, bgrmat);
                    //if (!viz_clouds->wasStopped())
                    //    viz_clouds->spin();
                }
            }
            else cout << "-->No Corners Detected on Image: " << data_dir_rgb_full << endl;
        
            corners_uv.clear();
            corners_real->points.clear();
            corners_align->points.clear();
            corners_ground_truth->points.clear();
        }
    }

    viz_clouds->close();
    destroyWindow("Corners Detection");
    dCP_terminate();
    DepthRMSE.close();
    DepthStdDevE.close();
    Distances.close();
    return 0;
}
