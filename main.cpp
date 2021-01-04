// 1. 读取一幅图像并显示
#if 0
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // 读取一幅图像
    Mat mat = imread("/home/user01/Pictures/1.jpg");
    namedWindow("testWindow");
    imshow("testWindow",mat);
    waitKey(0);
    return 0;
}
#endif


// 2. 创建一幅图像，采用ptr和at遍历像素
#if 0
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // 创建一幅矩阵， 600行， 800列， 灰度图
    Mat mat_gray_at(600,800,CV_8UC1);
    Mat mat_color_at(400,600,CV_8UC3);
    Mat mat_gray_ptr(600,800,CV_8UC1);
    Mat mat_color_ptr(400,600,CV_8UC3);

    // 遍历像素, 使用at, at为模板函数,
    for(int i = 0; i < mat_gray_at.rows; i++){
        for(int j = 0; j < mat_gray_at.cols; j++){
            uchar &g = mat_gray_at.at<uchar>(i,j);
            g = i * j / 50;
            //mat_gray.at<uchar>(i,j) = i * j / 50; // 也可以写成这个函数返回是一个引用
        }
    }
    for(int i = 0; i < mat_color_at.rows; i++){
        for(int j = 0; j < mat_color_at.cols; j++){
            Vec3b &c = mat_color_at.at<Vec3b>(i,j); // 这个函数返回是一个引用
            c[0] = 255;
            c[1] = 0;
            c[2] = 0;
        }
    }

    // 遍历像素, 使用数据指针ptr, ptr也为模板函数,
    for(int i = 0; i < mat_gray_ptr.rows; i++){
        for(int j = 0; j < mat_gray_ptr.cols; j++){
            *(mat_gray_ptr.ptr<uchar>(i,j)) = i * j / 50;
        }
    }
    for(int i = 0; i < mat_color_ptr.rows; i++){
        for(int j = 0; j < mat_color_ptr.cols; j++){
            Vec3b *c = mat_color_ptr.ptr<Vec3b>(i,j);
            c->val[0] = 0;
            c->val[1] = 255;
            c->val[2] = 0;
        }
    }

    imshow("gray_by_at",mat_gray_at);
    imshow("color_by_at",mat_color_at);
    imshow("gray_by_ptr",mat_gray_ptr);
    imshow("color_by_ptr",mat_color_ptr);
    waitKey(0);
    return 0;
}
#endif


// 3. 在一幅图像上添加文字
#if 0
// 	void cv::putText(
//		cv::Mat& img, // 待绘制的图像
//		const string& text, // 待绘制的文字
//		cv::Point origin, // 文本框的左下角
//		int fontFace, // 字体 (如cv::FONT_HERSHEY_PLAIN)
//		double fontScale, // 尺寸因子，值越大文字越大
//		cv::Scalar color, // 线条的颜色（RGB）
//		int thickness = 1, // 线条宽度
//		int lineType = 8, // 线型（4邻域或8邻域，默认8邻域）
//		bool bottomLeftOrigin = false // true='origin at lower left'
//	);
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

int main() {
    // 创建一幅矩阵， 400行， 600列， 灰度图
    Mat image(600,800,CV_8UC3);
    // 设置背景色
    image.setTo(cv::Scalar(100,0,0));
    // 通过设置像素来设置颜色
//    for(int i = 0; i < mat_color.rows; i++){
//        for(int j = 0; j < mat_color.cols; j++){
//            Vec3b &c = mat_color.at<Vec3b>(i,j); // 这个函数返回是一个引用
//            c[0] = 255;  // B
//            c[1] = 0;    // G
//            c[2] = 0;    // R
//        }
//    }

//设置绘制文本的相关参数
    std::string text = "Hello world!";         // 绘制文字
    cv::Point origin;                          // 文本框的左下角
    int font_face = cv::FONT_HERSHEY_COMPLEX;  // 字体
    double font_scale = 2;                     // 尺寸因子， 越大文字越大
    int thickness = 2;                         // 线条宽度
    int baseline;
    //获取文本框的长宽
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

    //将文本框居中绘制
    origin.x = image.cols / 2 - text_size.width / 2;
    origin.y = image.rows / 2 + text_size.height / 2;
    cv::putText(image, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

    //putText(mat_color,"hello world",Point(50,50),FONT_HERSHEY_PLAIN,2.0,Scalar(0,0,255),1,0);

    imshow("gray_by_at",image);
    waitKey(0);
    return 0;
}
#endif


// 4. use opencv headPose estimation
// using the picture provide by auther with 'headPose.jpg'
#if 0
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // Read input image
    cv::Mat im = cv::imread("headPose.jpg");

    // 2D image points. If you change the image, you need to change vector
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d(359, 391) );    // Nose tip
    image_points.push_back( cv::Point2d(399, 561) );    // Chin
    image_points.push_back( cv::Point2d(337, 297) );     // Left eye left corner
    image_points.push_back( cv::Point2d(513, 301) );    // Right eye right corner
    image_points.push_back( cv::Point2d(345, 465) );    // Left Mouth corner
    image_points.push_back( cv::Point2d(453, 469) );    // Right mouth corner

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner

    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

    cout << "Camera Matrix " << endl << camera_matrix << endl ;
    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);


    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose

    vector<Point3d> nose_end_point3D;
    vector<Point2d> nose_end_point2D;
    nose_end_point3D.push_back(Point3d(0,0,1000.0));

    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);


    for(int i=0; i < image_points.size(); i++)
    {
        circle(im, image_points[i], 3, Scalar(0,0,255), -1);
    }

    cv::line(im,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);

    cout << "Rotation Vector " << endl << rotation_vector << endl;
    cout << "Translation Vector" << endl << translation_vector << endl;

    cout <<  nose_end_point2D << endl;

    // Display image.
    cv::imshow("Output", im);
    cv::waitKey(0);
}
#endif


// 5. use openpose estimation human pose
#if 0
//  this sample demonstrates the use of pretrained openpose networks with opencv's dnn module.
//
//  it can be used for body pose detection, using either the COCO model(18 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
//
//  or the MPI model(16 parts):
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
//  https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt
//
//  (to simplify this sample, the body models are restricted to a single person.)
//
//  you can also try the hand pose model:
//  http://posefs1.perception.cs.cmu.edu/OpenPose/models/hand/pose_iter_102000.caffemodel
//  https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt
//
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
#include <iostream>
using namespace std;
// connection table, in the format [model_id][pair_id][from/to]
// please look at the nice explanation at the bottom of:
// https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
//
const int POSE_PAIRS[3][20][2] = {
        {   // COCO body
                {1,2}, {1,5}, {2,3},
                {3,4}, {5,6}, {6,7},
                {1,8}, {8,9}, {9,10},
                {1,11}, {11,12}, {12,13},
                {1,0}, {0,14},
                {14,16}, {0,15}, {15,17}
        },
        {   // MPI body
                {0,1}, {1,2}, {2,3},
                {3,4}, {1,5}, {5,6},
                {6,7}, {1,14}, {14,8}, {8,9},
                {9,10}, {14,11}, {11,12}, {12,13}
        },
        {   // hand
                {0,1}, {1,2}, {2,3}, {3,4},         // thumb
                {0,5}, {5,6}, {6,7}, {7,8},         // pinkie
                {0,9}, {9,10}, {10,11}, {11,12},    // middle
                {0,13}, {13,14}, {14,15}, {15,16},  // ring
                {0,17}, {17,18}, {18,19}, {19,20}   // small
        }};
int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{ h help | false     | print this help message }"
                             "{ p proto     | ../COCO/pose_coco.prototxt | (required) model configuration, e.g. hand/pose.prototxt }"
                             "{ m model     | ../COCO/pose_iter_440000.caffemodel | (required) model weights, e.g. hand/pose_iter_102000.caffemodel }"
                             "{ i image     | ../image/lin5.jpg | (required) path to image file (containing a single person, or hand) }"
                             "{ d dataset   |  COCO     | specify what kind of model was trained. It could be (COCO, MPI, HAND) depends on dataset. }"
                             "{ width       |  368      | Preprocess input image by resizing to a specific width. }"
                             "{ height      |  368      | Preprocess input image by resizing to a specific height. }"
                             "{ t threshold |  0.1      | threshold or confidence value for the heatmap }"
                             "{ s scale     |  0.003922 | scale for blob }"
    );
    String modelTxt = samples::findFile(parser.get<string>("proto"));
    String modelBin = samples::findFile(parser.get<string>("model"));
    String imageFile = samples::findFile(parser.get<String>("image"));
    String dataset = parser.get<String>("dataset");
    int W_in = parser.get<int>("width");
    int H_in = parser.get<int>("height");
    float thresh = parser.get<float>("threshold");
    float scale  = parser.get<float>("scale");
    if (parser.get<bool>("help") || modelTxt.empty() || modelBin.empty() || imageFile.empty())
    {
        cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << endl;
        parser.printMessage();
        return 0;
    }
    int midx, npairs, nparts;
    if (!dataset.compare("COCO")) {  midx = 0; npairs = 17; nparts = 18; }
    else if (!dataset.compare("MPI"))  {  midx = 1; npairs = 14; nparts = 16; }
    else if (!dataset.compare("HAND")) {  midx = 2; npairs = 20; nparts = 22; }
    else
    {
        std::cerr << "Can't interpret dataset parameter: " << dataset << std::endl;
        exit(-1);
    }
    // read the network model
    Net net = readNet(modelBin, modelTxt);
    // and the image
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
    // send it through the network
    Mat inputBlob = blobFromImage(img, scale, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
    net.setInput(inputBlob);
    Mat result = net.forward();
    // the result is an array of "heatmaps", the probability of a body part being in location x,y
    int H = result.size[2];
    int W = result.size[3];
    // find the position of the body parts
    vector<Point> points(22);
    for (int n=0; n<nparts; n++)
    {
        // Slice heatmap of corresponding body's part.
        Mat heatMap(H, W, CV_32F, result.ptr(0,n));
        // 1 maximum per heatmap
        Point p(-1,-1),pm;
        double conf;
        minMaxLoc(heatMap, 0, &conf, 0, &pm);
        if (conf > thresh)
            p = pm;
        points[n] = p;
    }
    // connect body parts and draw it !
    float SX = float(img.cols) / W;
    float SY = float(img.rows) / H;
    for (int n=0; n<npairs; n++)
    {
        // lookup 2 connected body/hand parts
        Point2f a = points[POSE_PAIRS[midx][n][0]];
        Point2f b = points[POSE_PAIRS[midx][n][1]];
        // we did not find enough confidence before
        if (a.x<=0 || a.y<=0 || b.x<=0 || b.y<=0)
            continue;
        // scale to image size
        a.x*=SX; a.y*=SY;
        b.x*=SX; b.y*=SY;
        line(img, a, b, Scalar(0,200,0), 2);
        circle(img, a, 3, Scalar(0,0,200), -1);
        circle(img, b, 3, Scalar(0,0,200), -1);
    }
    imshow("OpenPose", img);
    waitKey();
    return 0;
}
#endif


// 6. 字符分割 (1)方向投影法
// 暂未测试成功，2020/09/01
#if 0
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int vertical_projection(const Mat& src, vector<Mat>& roiList)
{
    //step1. 计算竖直投影白色点数量
    int w = src.cols;
    int h = src.rows;
    vector<int> project_val_arry;
    int per_pixel_value;
    for (int j=0;j<w;j++)//列
    {

        //int num = 0;
        //for (int i=0;i<h;i++)//行
        //{
        //	per_pixel_value = src.ptr<unsigned char>(i)[j];
        //	if (per_pixel_value == 255)
        //		num++;
        //}

        Mat j_im = src.col(j);
        int num = countNonZero(j_im);

        project_val_arry.push_back(num);
    }

    //显示
    if (1)
    {
        Mat hist_im(h, w, CV_8UC1, Scalar(255));
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < project_val_arry[i]; j++)
            {
                hist_im.ptr<unsigned char>(h - 1 - j)[i] = 0;
            }
        }
        imshow("project", hist_im);
        waitKey();
    }



    //step2. 字符分割
    //vector<Mat> roiList;
    int startIndex = 0;
    int endIndex = 0;
    bool inBlock = false;//是否遍历到了字符区内
    for (int i = 0; i < w; ++i)
    {
        if (!inBlock && project_val_arry[i] != 0)//进入字符区了
        {
            inBlock = true;
            startIndex = i;
            //cout << "startIndex is " << startIndex << endl;
        }
        else if (project_val_arry[i] == 0 && inBlock)//进入空白区了
        {
            endIndex = i;
            inBlock = false;
            Mat roiImg = src(Rect(startIndex,0,endIndex+1-startIndex,h));
            roiList.push_back(roiImg);
        }
    }

    return 0;
}

int main()
{
    Mat src = imread("../image/ocr01.png",0);
    Mat bin;
    threshold(src, bin, 60, 255, THRESH_OTSU);
    imshow("src", src);
    imshow("bin", bin);
    waitKey(0);
    vector<Mat> char_im_vec;
    vertical_projection(bin, char_im_vec);

    for (int i=0;i<char_im_vec.size();i++)
    {
        string win_name = "roi" + to_string(i);
        imshow(win_name, char_im_vec[i]);
    }

    waitKey(0);
    return 0;
}
#endif


// 7. 字符分割 1、图像单通道化，2、图像二值化；3、获取图像中的轮廓；4、实现分割。
#if 1
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

void SegmentChar(const string strPic)
{
	Mat img = imread(strPic, 0);
	if (!img.data)
	{
	    cout << "read image fail!" << endl;
		return;
	}
    imshow("src",img);

	Mat threshImg;
	threshold(img, threshImg, 100, 255, cv::THRESH_BINARY_INV); //图像的二值化
	std::vector<std::vector<Point>> contours;
	Mat hierarchy;
	findContours(threshImg, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
	drawContours(threshImg, contours, -1, Scalar(255, 0, 255), 1);
	std::vector<std::vector<Point>>::const_iterator iter = contours.begin();
	while (iter != contours.end())
	{
		Rect rc = boundingRect(*iter);
		rectangle(img, rc, Scalar(0, 255, 255), 1);
		iter++;
	}
	imshow("char", img);
	waitKey(0);
}

int main (){
    SegmentChar("../image/ocr02.png");
    print("hello\n");
    return 0;
}

#endif