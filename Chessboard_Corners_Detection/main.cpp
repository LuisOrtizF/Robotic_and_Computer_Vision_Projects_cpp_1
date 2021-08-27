#include <iostream>
#include <ctype.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

int main( )
{
    TermCriteria termcrit(TermCriteria::EPS|TermCriteria::EPS,100,0.01);
    Size patternsize(7,4); //interior number of corners
    Mat src = imread("../Images/test_0.png", IMREAD_COLOR );
    Mat gray; 
    cvtColor(src, gray, COLOR_BGR2GRAY); //source image
    vector<Point2f> corners; //this will be filled by the detected corners
    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, corners,
    CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);
    if(patternfound)
    cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), termcrit);
    
    drawChessboardCorners(src, patternsize, Mat(corners), patternfound);

    namedWindow("Chess", WINDOW_AUTOSIZE);

    imshow("Chess", src);

    waitKey(0);   
    return 0;
}