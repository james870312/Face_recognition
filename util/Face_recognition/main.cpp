#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>

#include <opencv2/opencv.hpp>

#define OPENCV
#include <yolo_v2_class.hpp>

using namespace std;
using namespace cv;

/** Collect class names from a file. */
std::vector<std::string> getClassName(std::string fileName) {
    std::ifstream file(fileName);
    std::vector<std::string> ret;
    std::string buf;
    while(std::getline(file, buf)) {
        ret.push_back(buf);
    }
    return ret;
}

/** Get an information of a bounding box. */
std::string boundingBoxInfo(bbox_t boundingBox) {
    char buf[50];
    std::sprintf(buf, "class %d: (%d, %d, %d, %d) = %.2lf %%",
            boundingBox.obj_id,
            boundingBox.x, boundingBox.y, boundingBox.w, boundingBox.h,
            boundingBox.prob*100);
    return string(buf);
}

/** Draw a bounding box onto a Mat, include drawing it's class name. */
void drawBoundingBox(cv::Mat image, bbox_t boundingBox, std::vector<std::string> classNames) {
    cv::Rect rect(boundingBox.x, boundingBox.y, boundingBox.w, boundingBox.h);
    //Random select a color of bounding box
    int r = 50 + ((43 * (boundingBox.obj_id + 1)) % 150);
    int g = 50 + ((97 * (boundingBox.obj_id + 1)) % 150);
    int b = 50 + ((37 * (boundingBox.obj_id + 1)) % 150);
    cv::Scalar color(b, g, r);
    cv::rectangle(image, rect, color, 2);
    cv::putText(image, classNames[boundingBox.obj_id], rect.tl() + cv::Point(0, 20), 0, 0.7, color, 2);
}


int main(int argc, char** argv) {

    //string imageFile = "/home/james/darknet/milk_dataset/jpg/Pic_18_12_27_20_47_25.jpg";

    string yoloCfgFile = "/home/james/Desktop/Computer_vision/HW3/image_files/mask_dataset/setting/yolov3-tiny.cfg";
    string yoloWeightFile = "/home/james/Desktop/Computer_vision/HW3/image_files/mask_dataset/model/yolov3-tiny_5300.weights";
    string yoloNameFile = "/home/james/Desktop/Computer_vision/HW3/image_files/mask_dataset/setting/obj.names";

    VideoCapture cam(0);
    if (!cam.isOpened())
    {
        cout << "Failed to open camera" << endl;
        return -1;
    }

    //Load YOLO model into memory
    Detector detector(yoloCfgFile, yoloWeightFile);
    //Load class names
    vector<string> classNames = getClassName(yoloNameFile);


    while(1)
    {
        //Mat image = imread(imageFile);
        Mat image;
        clock_t start, end;
        start = clock();

        bool ret = cam.read(image);
        if (!ret)
        {
            cout << "Failed to grab frame." << endl;
            break;
        }

        /*cout << "--------My classes-----" << endl;
        for(size_t i = 0; i < classNames.size(); i++) {
            cout << classNames[i] << endl;
        }
        cout << "-----------------------" << endl;*/

        //Predict an image
        vector<bbox_t> predict = detector.detect(image);

        for (size_t i = 0; i < predict.size(); i++) {
            cout << boundingBoxInfo(predict[i]) << endl;
            drawBoundingBox(image, predict[i], classNames);
            //cout <<"probability = "<< predict[i].prob*100 <<" %" << endl; 
        }


        end = clock();
        double fps = 1 / ((double) (end - start)) * CLOCKS_PER_SEC; 


        cout<<"fps = "<< fps <<endl;
        imshow("Camera", image);
        char kbin = waitKey(1);
        if (kbin == 27)
        {
            break;
        }
        //waitKey();
    }
    cam.release();
    destroyAllWindows();
    return 0;
}
