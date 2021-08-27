#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <dirent.h>
#include <iostream> 
#include <algorithm> 

using namespace std;
using namespace cv;

int main (int argc, char** argv)
{
	printf("\n%s\n\n", argv[0]);

	if (argc != 2)
	{
	    printf("\nUsage: %s <../Images>\n\n", argv[0]);
	    return 1;
	}

	string dir = argv[1];

	namedWindow("Image", CV_WINDOW_NORMAL);
	moveWindow("Image", 0, 0);
	resizeWindow("Image", 800, 400);

    int i = 0, k = 0;
    int file_count = 0;

    DIR * dirp;
    struct dirent * entry;

    dirp = opendir(dir.c_str()); /* There should be error handling after this */
    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type == DT_REG) { /* If the entry is a regular file */
            file_count++;
        }
    }
    closedir(dirp);

    vector<int> widths, heights;
    char base_name[256]; 
    string image_file_name;
    Mat image;

    while (i < file_count) 
    {
        // input file names
        sprintf(base_name,"%d.jpg", i);
        image_file_name  = dir + "/" + base_name;
        image = imread( image_file_name);
        
        // image dimensions
        int width_im  = image.cols;
        int height_im = image.rows;

        widths.push_back(width_im);
        heights.push_back(height_im);

        i++;
    }

    int min_w = *std::min_element(widths.begin(),widths.end());
    int min_h = *std::min_element(heights.begin(),heights.end());

    string dir2 = "../CroppedImages/";
    string save_file_name;

    while (k < file_count) 
    {
        // input file names
        sprintf(base_name,"%d.jpg", k);
        image_file_name  = dir + "/" + base_name;
        image = imread(image_file_name);

        int w  = image.cols;
        int h = image.rows;

        Rect r = Rect(w/2-min_w/2, h/2-min_h/2, min_w, min_h);
        
        Mat croppedImage = image(r);
        //ROI.copyTo(croppedImage);
        // cv::rectangle(image, Point(10,10), Point(100,100), Scalar(0,255,0),1,8,0);
        imshow("Image", croppedImage);
        save_file_name  =  dir2 + base_name;
        imwrite(save_file_name, croppedImage);
        waitKey(500);
        k++;
    }

    destroyAllWindows();
    return 0;
}