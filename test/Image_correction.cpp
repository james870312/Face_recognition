#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	//Load image
	const char* input = argv[1];
	Mat imageInput = imread(input, IMREAD_COLOR);

	if (imageInput.empty())
	{
		cout << "Failed to load " << input << endl;
		return -1;
	}

    Size board_size = Size(9,6);    /* 標定板上每行、列的角點數 */
    vector<Point2f> image_points_buf;  /* 快取每幅影象上檢測到的角點 */

    /* 提取角點 */
    if (0 == findChessboardCorners(imageInput,board_size,image_points_buf))
    {           
        cout<<"can not find chessboard corners!\n"; //找不到角點
        exit(1);
    } 
    else 
    {
        Mat view_gray;
        cvtColor(imageInput,view_gray,CV_RGB2GRAY);
        /* 亞畫素精確化 */
        find4QuadCornerSubpix(view_gray,image_points_buf,Size(20, 20)); //對粗提取的角點進行精確化
        //cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS CV_TERMCRIT_ITER,30,0.1));
        /* 在影象上顯示角點位置 */
        drawChessboardCorners(view_gray,board_size,image_points_buf,false); //用於在圖片中標記角點
        drawChessboardCorners(imageInput,board_size,image_points_buf,true); //用於在圖片中標記角點
        imshow("Camera Calibration_gray", view_gray);//顯示圖片
        imshow("Camera Calibration_color", imageInput);
        waitKey();
    }

    //printf("image_points = %f, %f ", image_points_buf[2].x, image_points_buf[2].y);

    Size image_size;  /* 影象的尺寸 */
    vector<vector<Point2f>> image_points_seq; /* 儲存檢測到的所有角點 */
    vector<vector<Point3f>> object_points; /* 儲存標定板上角點的三維座標 */
    /*內外引數*/
    Mat cameraMatrix=Mat(3,3,CV_32FC1,Scalar::all(0)); /* 攝像機內引數矩陣 */
    Mat distCoeffs=Mat(1,5,CV_32FC1,Scalar::all(0)); /* 攝像機的5個畸變係數：k1,k2,p1,p2,k3 */
    vector<Mat> tvecsMat;  /* 每幅影象的旋轉向量 */
    vector<Mat> rvecsMat; /* 每幅影象的平移向量 */

    image_points_seq.push_back(image_points_buf);  //儲存亞畫素角點
    image_size.width = imageInput.cols;//讀入第一張圖片時獲取影象寬高資訊
    image_size.height = imageInput.rows;

    /* 初始化標定板上角點的三維座標 */
    Size square_size = Size(10,10);  /* 實際測量得到的標定板上每個棋盤格的大小 */
    vector<Point3f> tempPointSet;
    for (int i=0; i<board_size.height; i++)
        {
            for (int j=0; j<board_size.width; j++)
                {
                    Point3f realPoint;
                    /* 假設標定板放在世界座標系中z=0的平面上 */
                    realPoint.x = i*square_size.width;
                    realPoint.y = j*square_size.height;
                    realPoint.z = 0;
                    tempPointSet.push_back(realPoint);
                }
        }
        object_points.push_back(tempPointSet);

    /* 開始標定 */
    calibrateCamera(object_points,image_points_seq,image_size,cameraMatrix,distCoeffs,rvecsMat,tvecsMat,0);

    //對標定結果進行評價
    //vector<Point3f> tempPointSet = object_points[0];
    vector<Point2f> image_points2; /* 儲存重新計算得到的投影點 */

    projectPoints(tempPointSet,rvecsMat[0],tvecsMat[0],cameraMatrix,distCoeffs,image_points2);

    //顯示定標結果
    cout<<"相機內引數矩陣："<<endl;   
    cout<<cameraMatrix<<endl<<endl;   
    cout<<"畸變係數：\n";   
    cout<<distCoeffs<<endl<<endl<<endl;

    //矯正影象

    Mat mapx = Mat(image_size,CV_32FC1);
    Mat mapy = Mat(image_size,CV_32FC1);
    Mat R = Mat::eye(3,3,CV_32F);

    initUndistortRectifyMap(cameraMatrix,distCoeffs,R,cameraMatrix,image_size,CV_32FC1,mapx,mapy);  

    remap(imageInput,imageInput,mapx, mapy, INTER_LINEAR);
    //undistort(imageInput,imageInput,cameraMatrix,distCoeffs);

    imshow("remap", imageInput);
	waitKey();
	destroyAllWindows();

	Mat left = imread("/home/james/Desktop/Computer_vision/HW3/image_files/Calibration_data/left13.png", IMREAD_GRAYSCALE);
    Mat right = imread("/home/james/Desktop/Computer_vision/HW3/image_files/Calibration_data/right13.png", IMREAD_GRAYSCALE);
    Mat disp;
    Rect leftROI, rightROI;

    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(0, 21);

    bm->setPreFilterType(CV_STEREO_BM_XSOBEL);  //CV_STEREO_BM_NORMALIZED_RESPONSE或者CV_STEREO_BM_XSOBEL
    bm->setPreFilterSize(9);
    bm->setPreFilterCap(31);
    bm->setBlockSize(15);
    bm->setMinDisparity(0);
    bm->setNumDisparities(64);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(5);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setROI1(leftROI);
    bm->setROI2(rightROI);

    bm->compute(left, right, disp);
    normalize(disp, disp, 0, 255, CV_MINMAX, CV_8U);

    imshow("left", left);
    imshow("right", right);
    imshow("dis", disp);
	waitKey();

	destroyAllWindows();

	return 0;
}
