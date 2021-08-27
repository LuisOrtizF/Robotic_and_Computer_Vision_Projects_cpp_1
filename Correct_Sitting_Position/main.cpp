#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <stdlib.h>
#include <iostream>
#include <pcl/console/time.h>

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_i, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_x, pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_icp, int dual)
{
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Visor-3D"));

      viewer->initCameraParameters();

      // Create two verticaly separated viewports
      int v1 (0);
      int v2 (1);
      viewer->createViewPort (0.0, 0.0, 0.5, 1.0, v1);
      viewer->createViewPort (0.5, 0.0, 1.0, 1.0, v2);

      if (dual==1){
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_i_color_h(cloud_i);
          viewer->addPointCloud(cloud_i, cloud_i_color_h, "cloud_i_v1", v1);
          viewer->addPointCloud(cloud_i, cloud_i_color_h, "cloud_i_v2", v2);

          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_x_color_h(cloud_x);
          viewer->addPointCloud(cloud_x, cloud_x_color_h, "cloud_x_v1", v1);

          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_icp_color_h(cloud_icp);
          viewer->addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);
      }
      else{
          // La nube de puntos Ideal es blanca
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_i_color_h(cloud_i, 255, 255, 255);
          viewer->addPointCloud(cloud_i, cloud_i_color_h, "cloud_i_v1", v1);
          viewer->addPointCloud(cloud_i, cloud_i_color_h, "cloud_i_v2", v2);

          // La nube de puntos X es verde
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_x_color_h(cloud_x, 20, 180, 20);
          viewer->addPointCloud(cloud_x, cloud_x_color_h, "cloud_x_v1", v1);

          // La nube de puntos resultado de ICP es roja
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_icp_color_h(cloud_icp, 180, 20, 20);
          viewer->addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

          // Adiciona texto a cada subventana
          viewer->addText ("White: initial point cloud.\nGreen: target point cloud'.", 10, 15, 16, 255, 255, 255, "icp_info_1", v1);
          viewer->addText ("White: initial point cloud'.\nRed: correct sitting position", 10, 15, 16, 255, 255, 255, "icp_info_2", v2);
      }

      // Configura el color de fondo de cada pantalla
      viewer->setBackgroundColor(0.5, 0.5, 0.5, v1);
      viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);

      return (viewer);
}

int main (int argc, char* argv[])
{
    if (argc != 4){
      std::cout << "Usage: './correct_position  <initial_cloud> <target_cloud> <mode_visualization>'\n";
      return (-1);
    }

    //Nube de puntos posicion Ideal
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_ideal (new pcl::PointCloud<pcl::PointXYZRGB>);
    //Nube de puntos posicion X
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr nube_x (new pcl::PointCloud<pcl::PointXYZRGB>);
    //Nube de puntos resultado de ICP
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr resultado (new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *nube_ideal) == -1)
    {
      PCL_ERROR ("Can't read file %s.\n", argv[1]);
      return (-1);
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[2], *nube_x) == -1)
    {
      PCL_ERROR ("Can't read file %s.\n", argv[2]);
      return (-1);
    }

    int modo;
    sscanf (argv[3],"%d",&modo);

    pcl::console::TicToc t_icp;
    t_icp.tic();

    //ALGORITMO ICP
    // <pcl::PointXYZRGB, pcl::PointXYZRGB> == <ptipo de datos de entrada, tipo de datos de salida>
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    // Configura las nubes de puntos:
    //      InputCloud (nube de puntos de entrada)=nube de puntos que queremos alinear para que se parezca a la nube d epuntos de destino
    //      InputTarget (nube de puntos de destino)
    icp.setInputCloud(nube_x);
    icp.setInputTarget(nube_ideal);

    // Configura el numero maximo de iteraciones
    icp.setMaximumIterations (1);
    // Configura el epsilon de transformacion
    //icp.setTransformationEpsilon (1e-8);
    // Configura la epsilon diferencia de distancia euclediana
    //icp.setEuclideanFitnessEpsilon (0.01);
    // COnfigura la distancia de correspondencia maxima (e.g. correspondencias altas distancias seran ignoradas)
    //icp.setMaxCorrespondenceDistance (0.05);
    // Use RANSAC to neglect false correspondences--1cm
    icp.setRANSACOutlierRejectionThreshold (0.01);

    // Alinea InputCloud a InputTarget
    // el resultado es una nube de puntos que es la copia transformada de la nube de entrada trasladada y rotada
    icp.align(*resultado);

    if (icp.hasConverged ())
    {
      // Obtiene la puntuación de desempeno Euclidiana (por ejemplo, suma de cuadrados de las distancias desde la fuente hasta el destino)
      std::cout << "ICP score: " << icp.getFitnessScore() << ".\n" << std::endl;
      // Obtiene la transformación que alinea InputCloud a InputTarget
      Eigen::Matrix4f transformation = icp.getFinalTransformation ();
      std::cout << "\nInput-Target transformation:\n" <<transformation << std::endl;
      cout<<"\nICP Time (1 iteraccion): "<< t_icp.toc ()<<"ms"<<endl;
      cout<<"initial cloud size: "<<nube_ideal->size()<< "\ntarget cloud size: "<< nube_x->size()<< "\nfinal cloud size: "<< resultado->size()<<endl;
    }
    else
    {
      PCL_ERROR ("ICP did not converge!\n");
      return (-1);
    }

   boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

   viewer = viewportsVis(nube_ideal, nube_x, resultado, modo);

   while (!viewer->wasStopped ())
   {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
   }
}
