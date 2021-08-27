#include <iostream>

// ZED includes
#include <sl/Camera.hpp>

// PCL includes
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

// Sample includes
#include <thread>
#include <mutex>

// Namespace
using namespace sl;
using namespace std;

// Global instance (ZED, Mat, callback)
Camera zed;
Mat data_cloud;
std::thread zed_callback;
std::mutex mutex_input;
bool stop_signal;
bool has_data;

// Sample functions
void startZED();
void run();
void closeZED();
void extractPlanes(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &, pcl::PointCloud<pcl::PointXYZ>::Ptr &);

shared_ptr<pcl::visualization::PCLVisualizer> viz_cloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr);
shared_ptr<pcl::visualization::PCLVisualizer> viz_plane(pcl::PointCloud<pcl::PointXYZ>::ConstPtr);

inline float convertColor(float colorIn);

// Main process

int main(int argc, char **argv) 
{
    if (argc > 2) 
    {
        cout << "Usage: ./extract_plane ../Data/test_0.svo" << endl;
        return -1;
    }

    // Set configuration parameters
    InitParameters init_params;
    if (argc == 2)
    {
        String input_path(argv[1]);
        init_params.input.setFromSVOFile(input_path);
    }
    else 
    {
        init_params.camera_resolution = RESOLUTION::HD2K;
        init_params.camera_fps = 60;
    }
    
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_mode = DEPTH_MODE::QUALITY;
    init_params.depth_minimum_distance = 0.30;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) 
    {
        cout << toString(err) << endl;
        zed.close();
        return 1;
    }

    int svo_position = 0;
    int nb_frames = 0;

    if (argc == 2)
        nb_frames = zed.getSVONumberOfFrames();

    // Allocate PCL point cloud at the resolution
    // Create the PCL point cloud visualizer

    sl::CameraInformation cam_info = zed.getCameraInformation();
    sl::Resolution resolution = cam_info.camera_configuration.resolution;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    p_pcl_point_cloud->points.resize(resolution.area());
    shared_ptr<pcl::visualization::PCLVisualizer> viewer = viz_cloud(p_pcl_point_cloud);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr planes(new pcl::PointCloud<pcl::PointXYZ>);
    shared_ptr<pcl::visualization::PCLVisualizer> viewer2 = viz_plane(planes);

    // Start ZED callback
    startZED();

    // Loop until viewer catches the stop signal
    while(!viewer->wasStopped()) 
    {
        // Try to lock the data if possible (not in use). Otherwise, do nothing.
        if (mutex_input.try_lock()) 
        {
            float *p_data_cloud = data_cloud.getPtr<float>();
            int index = 0;

            // Check and adjust points for PCL format
            for (auto &it : p_pcl_point_cloud->points) 
            {
                float X = p_data_cloud[index];
                if (!isValidMeasure(X)) // Checking if it's a valid point
                    it.x = it.y = it.z = it.rgb = 0;
                else 
                {
                    it.x = X;
                    it.y = p_data_cloud[index + 1];
                    it.z = p_data_cloud[index + 2];
                    it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
                }
                index += 4;
            }

            // Unlock data and update Point cloud
            mutex_input.unlock();
            viewer->updatePointCloud(p_pcl_point_cloud);
            viewer->spinOnce(1);

            extractPlanes(p_pcl_point_cloud, planes);
            viewer2->updatePointCloud(planes);
            viewer2->spinOnce(1);

            if (argc == 2){
                svo_position = zed.getSVOPosition();
                if (svo_position >= (nb_frames - 1))
                    break;
            }
            
        } 
        else
            sleep_ms(1);
    }

    // Close the viewer
    viewer->close();
    viewer2->close();

    // Close the zed
    closeZED();

    return 0;
}

/**
 *  This functions start the ZED's thread that grab images and data.
 **/
void startZED() 
{
    // Start the thread for grabbing ZED data
    stop_signal = false;
    has_data = false;
    zed_callback = std::thread(run);

    //Wait for data to be grabbed
    while (!has_data)
        sleep_ms(1);
}

/**
 *  This function loops to get the point cloud from the ZED. It can be considered as a callback.
 **/
void run() 
{
    while (!stop_signal) {
        if (zed.grab(SENSING_MODE::STANDARD) == ERROR_CODE::SUCCESS) {
            mutex_input.lock(); // To prevent from data corruption
            zed.retrieveMeasure(data_cloud, MEASURE::XYZRGBA);
            mutex_input.unlock();
            has_data = true;
        } else
            sleep_ms(1);
    }
}

/**
 *  This function frees and close the ZED, its callback(thread) and the viewer
 **/
void closeZED() 
{
    // Stop the thread
    stop_signal = true;
    zed_callback.join();
    zed.close();
}

/**
 *  This function creates a PCL visualizer
 **/
shared_ptr<pcl::visualization::PCLVisualizer> viz_cloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud) 
{
    // Open 3D viewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Cloud_Viewer"));
    
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);

    // Initial frame
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor (0.75, 0.75, 0.75);

    viewer->setShowFPS (false);	
    viewer->setSize(640,480);
    viewer->setPosition(0, 0);

    //Add axes labels
    pcl::PointXYZI point;
    point.getArray3fMap () << 1.1, 0, 0;
    viewer->addText3D ("X", point, 0.1, 1, 0, 0, "x_");
    point.getArray3fMap () << 0, 1.1, 0;
    viewer->addText3D ("Y", point, 0.1, 0, 1, 0, "y_");
    point.getArray3fMap () << 0, 0, 1.1;
    viewer->addText3D ("Z", point, 0.1, 0, 0, 1, "z_");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb);

    viewer->initCameraParameters();
    return (viewer);
}

shared_ptr<pcl::visualization::PCLVisualizer> viz_plane (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) 
{
    // Open 3D viewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Plane_Viewer"));
    
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);

    // Initial frame
    viewer->addCoordinateSystem(1.0);
    viewer->setBackgroundColor (0.75, 0.75, 0.75);

    viewer->setShowFPS (false);	
    viewer->setSize(640,480);
    viewer->setPosition(640+70, 0);

    //Add axes labels
    pcl::PointXYZI point;
    point.getArray3fMap () << 1.1, 0, 0;
    viewer->addText3D ("X", point, 0.1, 1, 0, 0, "x_");
    point.getArray3fMap () << 0, 1.1, 0;
    viewer->addText3D ("Y", point, 0.1, 0, 1, 0, "y_");
    point.getArray3fMap () << 0, 0, 1.1;
    viewer->addText3D ("Z", point, 0.1, 0, 0, 1, "z_");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, rgb);

    viewer->initCameraParameters();
    return (viewer);
}

/**
 *  This function convert a RGBA color packed into a packed RGBA PCL compatible format
 **/
inline float convertColor(float colorIn) 
{
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}

void extractPlanes( pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out)
{
    cerr << "PointCloud before filtering: " << cloud_in->size() << " data points." << endl;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_blob (new pcl::PointCloud<pcl::PointXYZ>), 
    cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>),
    cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

    pcl::copyPointCloud(*cloud_in, *cloud_blob);

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud_blob);
    sor.setLeafSize (0.01f, 0.01f, 0.01f);
    sor.filter (*cloud_filtered);
    cerr << "PointCloud after filtering: " << cloud_filtered->size() << " data points." << endl;

    // Segmentation
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    // Optional
    seg.setOptimizeCoefficients (true);
    
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.01);
    seg.setAxis (Eigen::Vector3f (0.0, 0.0, 1.0));
    seg.setEpsAngle (pcl::deg2rad(10.0));

    // Create the filtering object
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    int i = 0, nr_points = (int) cloud_filtered->points.size ();

    vector <pcl::PointCloud<pcl::PointXYZ>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZ>::Ptr > > sourceClouds;

    // While 30% of the original cloud is still there
    while (cloud_filtered->points.size () > 0.3 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_filtered);
        seg.segment (*inliers, *coefficients);

        if (inliers->indices.size () == 0)
        {
            cerr << "Could not estimate a planar model for the given dataset." << endl;
            break;
        }

        cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << endl;

        // Extract the inliers
        extract.setInputCloud (cloud_filtered);
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (*cloud_out);

        cerr << "PointCloud representing the planar component: " << cloud_out->size() << " data points." << endl;

        // std::stringstream ss;
        // ss << "table_scene_lms400_plane_" << i << ".pcd";
        // writer.write<pcl::PointXYZ> (ss.str (), *cloud_out, false);

        //save PointClouds to array
        sourceClouds.push_back(cloud_out);

        // Create the filtering object
        extract.setNegative (true);
        extract.filter (*cloud_f);
        cloud_filtered.swap (cloud_f);
        i++;
    }
}