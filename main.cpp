#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#pragma comment(lib, "opencv_world460d.lib")

void show_histogram(Mat src);

int main(){
    // VideoCapture cap("cat.mp4");
    // VideoCapture cap("flyman512x512.avi");
    VideoCapture cap(0, CAP_DSHOW);
    Mat BackGround = imread("flymanBG.jpg", IMREAD_GRAYSCALE);

    if (!cap.isOpened()) {
        cout << endl << "Error opening video file" << endl << endl;
        return -1;
    }

    namedWindow("Video Player");

    // Main parameter
    Mat frame, WebCamBackGround;
    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 Background subtractor
    pMOG2->setHistory(20);
    pMOG2->setVarThreshold(100);
    pMOG2->setDetectShadows(false);

    char key = 0;
    int mode[8] = { 1, 0, 0, 0, 0, 0, 0, 0 };
    while (key != 'q') {
        // Read a frame from the video file
        cap >> frame;

        if (frame.empty()) {
            cout << "End of video file" << endl;
            break;
        }

        // function mode
        if (key == 'n' || mode[0] == 1) {       // 1, 0, 0, 0, 0, 0, 0, 0
            mode[0] = 1;
            for (int i = 1; i < 6; i++) {
                mode[i] = 0;
            }
        }
        if (key == 'g' || mode[1] == 1) {     // 切換為灰階影像
            mode[0] = 0;
            if (mode[1] == 0 && key == 'g') mode[1] = 1;
            else if (key == 'g') mode[1] = 0;

            cvtColor(frame, frame, COLOR_BGR2GRAY); // Gray scale

        }
        if (key == 'i' || mode[2] == 1) {       // 負片   
            mode[0] = 0;
            if (mode[2] == 0 && key == 'i') mode[2] = 1;
            else if (key == 'i') mode[2] = 0;

            frame = 255 - frame; // Complement
        }
        if (key == 'c') {                     // 抓取背景影像 (WebCam)
            cvtColor(frame, frame, COLOR_BGR2GRAY);
            WebCamBackGround = frame;
        }
        if (key == 'f' || mode[4] == 1) {     // 計算前景並顯示 (WebCam)
            mode[0] = 0;
            if (mode[4] == 0 && key == 'f') mode[4] = 1;
            else if (key == 'f') mode[4] = 0;

            if (WebCamBackGround.empty()) {
                cout << "No webcam background" << endl;
                break;
            }
            cvtColor(frame, frame, COLOR_BGR2GRAY); 
            absdiff(frame, WebCamBackGround, frame);
        }
        if (key == 'b' || mode[5] == 1) {     // Background subtraction
            mode[0] = 0;
            if (mode[5] == 0 && key == 'b') mode[5] = 1;
            else if (key == 'b') mode[5] = 0;

            //create Background Subtractor objects
            pMOG2->apply(frame, frame);

        }
        if (key == 't' || mode[6] == 1) {     // Binary thresholding
            if (mode[6] == 0 && key == 't') mode[6] = 1;
            else if (key == 't') mode[6] = 0;
                
            cvtColor(frame, frame, COLOR_BGR2GRAY); // Gray scale
            absdiff(frame, BackGround, frame);
            threshold(frame, frame, 32, 255, THRESH_BINARY);
        }                           
        if (key == 'h' || mode[7] == 1) {     // Histogram
            if (mode[7] == 0 && key == 'h') mode[7] = 1;
            else if (key == 'h') mode[7] = 0;

            show_histogram(frame);
        }

        imshow("Video Player", frame);
        key = waitKey(30);
    }

    // Release the video file and destroy the window
    cap.release();
    destroyWindow("Video Player");

    return 0;
}

void show_histogram(Mat src) {
    int bins = 256;
    int hist_size[] = { bins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    int channels[] = { 0 };

    Mat hist;
    calcHist(&src, 1, channels, Mat(), hist, 1, hist_size, ranges, true, false);

    // plot histogram
    double maxVal;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    int scale = 2;
    int hist_height = 256;
    Mat hist_img = Mat::zeros(hist_height, bins * scale, CV_8UC3);
    for (int i = 0; i < bins; i++) {
        float binVal = hist.at<float>(i);
        int intensity = cvRound(binVal / maxVal * hist_height);
        rectangle(hist_img, Point(i * scale, hist_height - 1),
            Point((i + 1) * scale - 1, hist_height - intensity), CV_RGB(255, 255, 255));
    }

    imshow("histogram", hist_img);
    waitKey(30);
}