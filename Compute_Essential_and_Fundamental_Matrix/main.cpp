
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp> 

#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
using namespace Eigen;


int main( int argc, char** argv )	
{
  
  //read the first two frames from the dataset
  Mat left, right;
  left = imread("../Data/left.png");
  right = imread("../Data/right.png");

  if ( !left.data || !right.data ) { 
    std::cout<< " --(!) Error reading images " << std::endl; return -1;
  }

  double focal = 707.0912;
  Point2d pp(601.8873, 183.1104);

  int minHessian = 400;
  Ptr<SURF> detector = SURF::create( minHessian );
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;

  std::vector<Point2f> vo_left, vo_rigth;
  std::vector<KeyPoint> left_keypoints, rigth_keypoints;
  Mat left_descriptors, rigth_descriptors;

  detector->detectAndCompute( left, Mat(), left_keypoints, left_descriptors );
  left_descriptors.convertTo(left_descriptors, CV_32F);

  detector->detectAndCompute( right, Mat(), rigth_keypoints, rigth_descriptors );
  rigth_descriptors.convertTo(rigth_descriptors, CV_32F);

  matcher.match( left_descriptors, rigth_descriptors, matches );

  double min_dist = 0.05;

  std::vector< DMatch > good_matches;

  for( int i = 0; i < left_descriptors.rows; i++ )
    if( matches[i].distance <= 3*min_dist )
        good_matches.push_back( matches[i]);

  vector<int> v2;

  for (int i = 0; i < good_matches.size(); i++)
    v2.push_back(good_matches[i].trainIdx);

  // holds count of each encountered number
  unordered_map<int, size_t> count;   
  
  for (int i = 0; i < v2.size(); i++)        
    count[v2[i]]++;             

  int aux = 0;

  std::vector< DMatch > good_matches2;

  for (auto &e:count)
  {
    for (int i = 0; i < good_matches.size(); i++)
    {
      if (e.first == good_matches[i].trainIdx)
      {
        aux++;
        if (aux == 1)
          good_matches2.push_back(good_matches[i]);
      }
    }
    aux = 0;
  }

  // cout << count.size() << endl;
  // cout << good_matches2.size() << endl;
    
  for( size_t i = 0; i < good_matches2.size(); i++ )
  {
    vo_left.push_back( left_keypoints[ good_matches2[i].queryIdx ].pt );
    vo_rigth.push_back( rigth_keypoints[ good_matches2[i].trainIdx ].pt );
  }

  Mat E, R1, t1, mask1;
  
  E = findEssentialMat(vo_rigth, vo_left, focal, pp, RANSAC, 0.999, 1.0, mask1); // Nister Solution

  cout << "\nEssential Matrix: " << E << endl;

  recoverPose(E, vo_rigth, vo_left, R1, t1, focal, pp, mask1);

  Mat t41 = (Mat_<double>(3,1)<<-0.5370000, 0.004822061, -0.01252488);
  
  Eigen::Vector3d t_eg;

  cv2eigen(t41, t_eg);

  double scale = t_eg.norm();

  //cout<< t1*scale << endl;

  Mat F;

  Mat K = (Mat_<double>(3,3)<< 707.0912, 0, 601.8873, 0, 707.0912, 183.1104, 0, 0, 1);

  sfm::fundamentalFromEssential	(	E, K, K, F);

  cout << "\nFundamental Matrix: " << F << endl;

  //Devuelve la matriz fundamental normalizada
  Mat F2 = findFundamentalMat(vo_rigth, vo_left, CV_FM_RANSAC, 3, 0.99);

  cout<< "\nNormalized Fundamental Matrix: " << F2 << endl;

  // namedWindow( "left", WINDOW_AUTOSIZE );// Create a window for display.
  // namedWindow( "right", WINDOW_AUTOSIZE );// Create a window for display.

  // imshow( "left", left );
  // imshow( "right", right );

    //-- Draw matches
  Mat img_matches;
  drawMatches( left, left_keypoints, right, rigth_keypoints, good_matches2, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow("Good Matches", img_matches );

  good_matches.clear();
  good_matches2.clear();
  vo_left.clear();
  vo_rigth.clear();
  
  waitKey();
  
  return 0;
}
