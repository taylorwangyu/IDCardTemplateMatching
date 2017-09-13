//
//  main.cpp
//  LocateByHaar
//
//  Created by Taylor on 18/07/2017.
//  Copyright © 2017 YW. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;
//Fixed size for a Character
const int H = 19;
const int W = 16;

struct fPos{
    CvPoint p;
    double feature;
};

int findCharacter(CvPoint pos, vector<CvPoint> characters, vector<bool> already)
{
    //min Distance to locate a character
    float minDis = 0.55*sqrt(W*W+H*H);
    int ans = -1;
    for (int i = 0; i < characters.size(); i++)
        if (!already[i])
        {
            float Dis = sqrt((characters[i].x - pos.x)*(characters[i].x - pos.x)+(characters[i].y - pos.y)*(characters[i].y - pos.y));
            //Find the closest one
            if (Dis < minDis)
            {
                minDis = Dis;
                ans = i;
            }
        }
    
    return ans;
}

bool check(int i, vector<CvPoint> characters, vector<int>& ans)
{
    vector<bool> already(characters.size(), false);
    CvPoint cha_xing, cha_ming, cha_xing1, cha_bie, cha_chu, cha_sheng, cha_di, cha_zhi,cha_min, cha_zu, cha_nian, cha_yue,cha_ri;
    int ret;
    //Suppose the character i is xing
    cha_xing = characters[i];
    already.push_back(i);
    ans.push_back(i);
    
    //Find the character ming
    ret = findCharacter(cvPoint(cha_xing.x, cha_xing.y + 36), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_ming = characters[ret];
    }
    
    //Calculate the fixed distance by xing and ming
    double std = cha_ming.y - cha_xing.y;
    double std_h = 50.0 * std / 36.0;
    double std_w1 = 118.0 *std / 36.0;
    double std_w2 = 66.0 *std / 36.0;
    
    
    //Find the character xing1
    ret = findCharacter(cvPoint(cha_xing.x + std_h, cha_xing.y), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_xing1 = characters[ret];
    }
    //Find the character bie
    ret = findCharacter(cvPoint(cha_xing1.x, cha_xing1.y + std), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_bie = characters[ret];
    }
    //Find the character chu
    ret = findCharacter(cvPoint(cha_xing1.x + std_h, cha_xing1.y), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_chu = characters[ret];
    }
    //Find the character sheng
    ret = findCharacter(cvPoint(cha_chu.x, cha_chu.y + std), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_sheng = characters[ret];
    }
    //Find the character di
    ret = findCharacter(cvPoint(cha_chu.x + std_h, cha_chu.y), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_di = characters[ret];
    }
    //Find the character zhi
    ret = findCharacter(cvPoint(cha_di.x, cha_di.y + std), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_zhi = characters[ret];
    }
    //Find the character min
    ret = findCharacter(cvPoint(cha_bie.x, cha_bie.y + std_w1), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_min = characters[ret];
    }
    //Find the character zu
    ret = findCharacter(cvPoint(cha_min.x, cha_min.y + std), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_zu = characters[ret];
    }
    ////Find the character nian
    ret = findCharacter(cvPoint(cha_min.x + std_h, cha_min.y), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_nian = characters[ret];
    }
    //Find the character yue
    ret = findCharacter(cvPoint(cha_nian.x, cha_nian.y + std_w2), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_yue = characters[ret];
    }
    //Find the character ri
    ret = findCharacter(cvPoint(cha_yue.x, cha_yue.y + std_w2), characters, already);
    if (ret == -1)
        return false;
    else{
        already.push_back(ret);
        ans.push_back(ret);
        cha_ri = characters[ret];
    }
    
    return true;
}

//Compare function for feature sort
bool Compare(const fPos& a, const fPos& b)
{
    return a.feature > b.feature;
}

//Compare function for axis Y sort
bool posCompare(const fPos& a, const fPos& b)
{
    return a.p.y > b.p.y;
}

//Function for determine whether two characters are overlapped
bool isOverlap(const CvPoint a, const CvPoint b)
{
    return ((abs(a.x-b.x)<=H)&&(abs(a.y-b.y)<=W));
}

void featureExtract(vector<CvPoint>& cleanTarget, vector<double>& features, Mat img)
{
    int rows = img.rows;
    int cols = img.cols;
    //Calculate the Integral Image
    vector<vector<double> > s(rows, vector<double >(cols));
    vector<vector<double> > ii(rows, vector<double >(cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == 0)
                s[i][j] = img.at<uchar>(i,j);
            else
                s[i][j] = img.at<uchar>(i,j) + s[i][j-1];
            if (i == 0)
                ii[i][j] = s[i][j];
            else
                ii[i][j] = s[i][j] + ii[i-1][j];
        }
    }
    
    //Calculate Haar-like Feature for all of the pixels in the image
    vector<fPos> target;
    vector<bool> use;
    for (int x = 0; x < rows - H; x++) {
        for (int y = 0; y < cols -2*W; y++) {
            double rec1 = ii[x+H][y+2*W] - ii[x][y+2*W] - ii[x+H][y] + ii[x][y];
            double rec2 = ii[x+H][y+W*3/2] - ii[x][y+W*3/2] - ii[x+H][y+W/2] + ii[x][y+W/2];
            double feature = rec1 - 2*rec2;
            //When the Feature is greater than the threshold, We recognize it as a potential character
            if (feature > 3600)
            {
                struct fPos f = *new struct fPos;
                f.p = cvPoint(x, y+W/2);
                f.feature = feature;
                target.push_back(f);
                use.push_back(false);
            }
        }
    }
    //sort the potential characters by feature from big to small
    sort(target.begin(), target.end(), Compare);
    
    //clean the Overlapped characters
    vector<fPos> temp;
    for (int i = 0; i < target.size(); i++)
        if (!use[i])
        {
            use[i] = true;
            for (int j = i+1; j < target.size(); j++)
                if (isOverlap(target[i].p, target[j].p))
                    use[j] = true;
            temp.push_back(target[i]);
        }
    
    //sort the potential characters by axis Y from big to small
    sort(temp.begin(), temp.end(), posCompare);
    for (int i = 0; i < temp.size(); i++){
        cleanTarget.push_back(temp[i].p);
        features.push_back(temp[i].feature);
    }
    
}

bool Matching(Mat img, vector<CvPoint>& result)
{
    vector<CvPoint> cleanTarget;
    vector<double> features;
    Mat gray;
    Mat tmp_m, tmp_sd;
    double mean, std;
    
    int original_cols = img.cols;
    int original_rows = img.rows;
    
    //Resize the img and change it into gray img
    resize(img, img, Size(714,444));
    cvtColor(img, gray, CV_BGR2GRAY);
    
    //Calculate the mean and standard deviation in the ROI
    Mat grayROI = gray(Rect(50,50,gray.cols/2,gray.rows-50));
    meanStdDev(grayROI, tmp_m, tmp_sd);
    mean = tmp_m.at<double>(0,0);
    std = tmp_sd.at<double>(0,0);
    
    //Manipulate the gray img to the fixed mean(128) and standard deviation(45)
    gray = (gray - mean) / std * 45 + 128;
    GaussianBlur(gray, gray, Size(5, 5), 0, 0);
    
    //Find postitions for all of the potential characters
    featureExtract(cleanTarget, features, gray);
    
    //Find the fixed templet in these characters
    vector<int> ans;
    result.clear();
    
    for (int i = 0; i < cleanTarget.size(); i++){
        if ((cleanTarget[i].y > gray.cols/4)||(cleanTarget[i].x > gray.rows/3))
            continue;
        
        int ret = check(i, cleanTarget, ans);
        //Find the solution match the templete
        if (ret == true)
        {
            for (int j = 0; j < ans.size(); j++){
                CvPoint temp = cleanTarget[ans[j]];
                int y = int(double(temp.y) / 714.0 * double(original_cols));
                int x = int(double(temp.x) / 444.0 * double(original_rows));
                result.push_back(cvPoint(x, y));
            }
            return true;
        }
        ans.clear();
    }
    
    return false;
}

//Rotate the ucmatImg by dDegree
Mat ImgRotate(const Mat& ucmatImg, double dDegree)
{
    Mat ucImgRotate;
    
    double a = sin(dDegree  * CV_PI / 180);
    double b = cos(dDegree  * CV_PI / 180);
    int width = ucmatImg.cols;
    int height = ucmatImg.rows;
    int width_rotate = int(height * fabs(a) + width * fabs(b));
    int height_rotate = int(width * fabs(a) + height * fabs(b));
    
    Point center = Point(ucmatImg.cols / 2, ucmatImg.rows / 2);
    
    Mat map_matrix = getRotationMatrix2D(center, dDegree, 1.0);
    map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;
    map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;
    
    warpAffine(ucmatImg, ucImgRotate, map_matrix, { width_rotate, height_rotate },
               CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT, cvScalarAll(0));
    
    return ucImgRotate;
}

//Rotate the Img and return the result by the order of (姓名性别出生地址民族年月日)
bool MatchingFor2Direction(Mat img, vector<CvPoint>& result){
    //Img not exist
    if (img.cols*img.rows == 0)
        return false;
    
    //When the img's height larger than width
    if (img.rows > img.cols){
        //Rotate the img by 90 degrees clockwise
        Mat imgCW90 = ImgRotate(img, -90);
        if (Matching(imgCW90, result))
            return true;
        else{
            //Rotate the img by 90 degree anticlockwise
            Mat imgACW90 = ImgRotate(img, 90);
            if (Matching(imgACW90, result))
                return true;
            else
                return false; //no Matching
        }
    }
    else{     //When the img's width larger than height
        //No Rotation
        Mat img0 = img;
        if (Matching(img0, result))
            return true;
        else{
            //Rotate the img by 180 degrees
            Mat img180 = ImgRotate(img, 180);
            if (Matching(img180, result))
                return true;
            else
                return false; //no Matching
        }
    }
    
}

int main(int argc, const char * argv[]) {
    struct dirent *dirp;
    
    string path ="/Users/Taylor/Lianlian/idcard/Normalidcard/";
    DIR* dir = opendir(path.c_str());
    
    int count = 0;
    while ((dirp = readdir(dir)) != NULL) {
        if (dirp->d_type == DT_REG) {
            // 文件
            //printf("%s\n", dirp->d_name);
            string filename = dirp->d_name;
            filename = path + filename;
            Mat img = imread(filename);
            vector<CvPoint> result;
            bool ret = MatchingFor2Direction(img, result);
            if (!ret)
                cout<<count<<":Fail!"<<endl;
            else
                cout<<count<<":Success!"<<endl;
        } else if (dirp->d_type == DT_DIR) {
            // 文件夹
        }
        count++;
    }
    
    closedir(dir);
    
    return 0;
}
