#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//Load image
	const char* input = argv[1];
	Mat src = imread(input, IMREAD_COLOR);

	if (src.empty())
	{
		cout << "Failed to load " << input << endl;
		return -1;
	}

    //set parameter
    float sigma = 1;
    int kernel = 3;
    float sum = 0;

	//Create gaussian kernel
	Mat G_kernel(kernel, kernel, CV_32F, Scalar::all(0));
    for(int i=0; i<kernel; i++)
    {
        for(int j=0; j<kernel; j++)
        {
            G_kernel.at<float>(i,j) = exp((-1)*(pow((i)-int(kernel/2), 2)+pow((j)-int(kernel/2), 2))/2*pow(sigma, 2));
            sum = sum + G_kernel.at<float>(i,j);
        }
    }

    //creat Gray Mat
    Mat gray(src.rows, src.cols, CV_8U);
    cout << "rows = " << src.rows << endl;
    cout << "cols = " << src.cols << endl;

    //RGB to Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue
    for(int i=0; i<src.rows; i++)
    {
        for(int j=0; j<src.cols; j++)
        {
            gray.at<uchar>(i, j) = 0.114*src.at<uchar>(i, 3*j) 
                                    + 0.587*src.at<uchar>(i, 3*j+1) 
                                    + 0.299*src.at<uchar>(i, 3*j+2);
        }
    }

    //print blur kernel
    cout << "blur kernel:" << endl;
    for(int i=0; i<kernel; i++)
    {
        for(int j=0; j<kernel; j++)
        {
        cout << G_kernel.at<float>(i, j) << "\t" ;
        }
        cout << endl;
    }

    //convolution operation with Gaussian kernel
    Mat blur(gray.rows-2, gray.cols-2, CV_8U);

    for(int i=0; i<gray.rows-2; i++)
    {
        for(int j=0; j<gray.cols-2; j++)
        {
            int cal = 0;
            for(int k=0; k<kernel; k++)
            {
                for(int n=0; n<kernel; n++)
                {
                    cal = cal + G_kernel.at<float>(k, n)*gray.at<uchar>(i+k, j+n);
                }
            }
            blur.at<uchar>(i, j) = cal/sum;
        }
    }

    //Create sobel kernel
    Mat Gx_kernel(3, 3, CV_32F, Scalar::all(0));
    Gx_kernel.at<float>(0, 0) = -1.0;
    Gx_kernel.at<float>(1, 0) = -2.0;
    Gx_kernel.at<float>(2, 0) = -1.0;
    Gx_kernel.at<float>(0, 2) =  1.0;
    Gx_kernel.at<float>(1, 2) =  2.0;
    Gx_kernel.at<float>(2, 2) =  1.0;

    Mat Gy_kernel(3, 3, CV_32F, Scalar::all(0));
    Gy_kernel.at<float>(0, 0) = -1.0;
    Gy_kernel.at<float>(0, 1) = -2.0;
    Gy_kernel.at<float>(0, 2) = -1.0;
    Gy_kernel.at<float>(2, 0) =  1.0;
    Gy_kernel.at<float>(2, 1) =  2.0;
    Gy_kernel.at<float>(2, 2) =  1.0;
    
    //print Gx kernel
    cout << "Gx kernel:" << endl;
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
        cout << Gx_kernel.at<float>(i, j) << "\t" ;
        }
        cout << endl;
    }
    //print Gy kernel
    cout << "Gy kernel:" << endl;
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
        cout << Gy_kernel.at<float>(i, j) << "\t" ;
        }
        cout << endl;
    }
    
    //set parameter
    Mat Gx(gray.rows-4, gray.cols-4, CV_8U);
    Mat Gy(gray.rows-4, gray.cols-4, CV_8U);
    Mat G(gray.rows-4, gray.cols-4, CV_8U);
    Mat non_G(gray.rows-4, gray.cols-4, CV_8U);
    Mat canny(gray.rows-4, gray.cols-4, CV_8U);
    Mat result (src.rows, src.cols, CV_8UC3);
    float gradient[gray.rows-4][gray.cols -4] = {0};
    float angle[gray.rows-4][gray.cols -4] = {0};

    //convolution operation with Gx & Gy kernel
    //find the intensity and angle gradient of the image
    for(int i=0; i<gray.rows-4; i++)
    {
        for(int j=0; j<gray.cols-4; j++)
        {
            float cal = 0;
            float cal2 = 0;
            for(int k=0; k<3; k++)
            {
                for(int n=0; n<3; n++)
                {
                    cal = cal + Gx_kernel.at<float>(k, n)*gray.at<uchar>(i+k, j+n);
                    cal2 = cal2 + Gy_kernel.at<float>(k, n)*gray.at<uchar>(i+k, j+n);
                }
            }
            if(cal<0){cal = 0;}
            else if(cal>255){cal = 255;}
            if(cal2<0){cal2 = 0;}
            else if(cal2>255){cal2 = 255;}
            Gx.at<uchar>(i, j) = cal;
            Gy.at<uchar>(i, j) = cal2;
            gradient[i][j] = sqrt(pow(cal, 2)+pow(cal2, 2));
            if(gradient[i][j]>255){gradient[i][j] = 255;}

            if(cal==0) {angle[i][j] = 90;}
            angle[i][j] = atan(cal2/cal)*180/3.14;
            if(angle[i][j]< 22.5 && angle[i][j]>-22.5){angle[i][j] = 0;}
            else if(angle[i][j]< 67.5 && angle[i][j]> 22.5){angle[i][j] = 45;}
            else if(angle[i][j]<-22.5 && angle[i][j]>-67.5){angle[i][j] = 135;}
            else if(angle[i][j]<-67.5 || angle[i][j]> 67.5){angle[i][j] = 90;}
            G.at<uchar>(i, j) = gradient[i][j];
            if(gradient[i][j]>255){G.at<uchar>(i, j)=255;}
        }
    }
    
    //Non-maximum suppression
    for(int i=1; i<G.rows-1; i++)
    {
        for(int j=1; j<G.cols-1; j++)
        {
            if(angle[i][j]==0)
            {
                if(gradient[i][j]<gradient[i][j-1]){gradient[i][j]=0;}
                else{gradient[i][j-1]=0;}
                if(gradient[i][j]<gradient[i][j+1]){gradient[i][j]=0;}
                else{gradient[i][j+1]=0;}
                if(gradient[i][j+1]<gradient[i][j-1]){gradient[i][j+1]=0;}
                else{gradient[i][j-1]=0;}
            }
            else if(angle[i][j]==45)
            {
                if(gradient[i][j]<gradient[i+1][j-1]){gradient[i][j]=0;}
                else{gradient[i+1][j-1]=0;}
                if(gradient[i][j]<gradient[i-1][j+1]){gradient[i][j]=0;}
                else{gradient[i-1][j+1]=0;}
                if(gradient[i+1][j-1]<gradient[i-1][j+1]){gradient[i+1][j-1]=0;}
                else{gradient[i-1][j+1]=0;}
            }
            else if(angle[i][j]==90)
            {
                if(gradient[i][j]<gradient[i-1][j]){gradient[i][j]=0;}
                else{gradient[i-1][j]=0;}
                if(gradient[i][j]<gradient[i+1][j]){gradient[i][j]=0;}
                else{gradient[i+1][j]=0;}
                if(gradient[i+1][j]<gradient[i-1][j]){gradient[i+1][j]=0;}
                else{gradient[i-1][j]=0;}
            }
            else if(angle[i][j]==135)
            {
                if(gradient[i][j]<gradient[i-1][j-1]){gradient[i][j]=0;}
                else{gradient[i-1][j-1]=0;}
                if(gradient[i][j]<gradient[i+1][j+1]){gradient[i][j]=0;}
                else{gradient[i+1][j+1]=0;}
                if(gradient[i+1][j+1]<gradient[i-1][j-1]){gradient[i+1][j+1]=0;}
                else{gradient[i-1][j-1]=0;}
            }
            non_G.at<uchar>(i, j) = gradient[i][j];
            if(gradient[i][j]>255){non_G.at<uchar>(i, j)=255;}
        }
    }
    
    //set paramete
    int H_thresd = 100;
    int L_thresd = 100;

    namedWindow("tweak");
    createTrackbar("H_thres", "tweak", &H_thresd, 256);
    createTrackbar("L_thres2", "tweak", &L_thresd, 256);

    //Connect Weak Edge & draw on source image
    while(1)
    {    
        for(int i=0; i<non_G.rows; i++)
        {
            for(int j=0; j<non_G.cols; j++)
            {
                result.at<uchar>(i+2, 3*(j+2)) = src.at<uchar>(i+2, 3*(j+2));
                result.at<uchar>(i+2, 3*(j+2)+1) = src.at<uchar>(i+2, 3*(j+2)+1);
                result.at<uchar>(i+2, 3*(j+2)+2) = src.at<uchar>(i+2, 3*(j+2)+2);
                canny.at<uchar>(i, j) = 0;
                if(non_G.at<uchar>(i, j) > H_thresd)
                {
                    canny.at<uchar>(i,j) = 255;
                    result.at<uchar>(i+2, 3*(j+2)) = 0;               
                    result.at<uchar>(i+2, 3*(j+2)+1) = 255;               
                    result.at<uchar>(i+2, 3*(j+2)+2) = 0;               
                }
                else if(non_G.at<uchar>(i, j) < L_thresd){canny.at<uchar>(i,j) = 0;}
                else
                {
                    for(int k=-1; k<2; k++)
                    {
                        for(int n=-1; n<2; n++)
                        {
                            //if(non_G.at<uchar>(i+k, j+n) > H_thresd)
                            if(canny.at<uchar>(i+k, j+n) ==255)
                            {
                                canny.at<uchar>(i, j) = 255;
                                result.at<uchar>(i+2, 3*(j+2)) = 0;               
                                result.at<uchar>(i+2, 3*(j+2)+1) = 255;               
                                result.at<uchar>(i+2, 3*(j+2)+2) = 0;               
                            }
                        }
                    }
                }
            }
        }

    	//Show result
    	//imshow("input", src);
    	//imshow("Gray", gray);
        //imshow("blur", blur);
        imshow("Gx", Gx);
        imshow("Gy", Gy);
        imshow("G", G);
        //imshow("NON_Maximal", non_G);
        imshow("canny", canny);
        imshow("result", result);
        
        char kbin = waitKey(30);
        if(kbin == 27){break;}
    }

	destroyAllWindows();

	return 0;
}
