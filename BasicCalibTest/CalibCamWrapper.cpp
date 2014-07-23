#include "stdafx.h"

#define _USE_MATH_DEFINES

#include <opencv2\opencv.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <vector>

__declspec(dllexport)
double CalibCameraWrapper(
    cv::Point3f const * pts3d, cv::Point2f const * pts2d, unsigned int ptCount,
    unsigned int imgWidth, unsigned int imgHeight, float * intrinsicMatInOut,
    float * rotationOut, float * translationOut )
{
    cv::Mat distCoeffs_est;
    std::vector<cv::Mat> rvecs(1), tvecs(1);

    cv::Mat1d camMat_est(3,3, 0.);
    if( intrinsicMatInOut )
    {
        cv::Mat1f(3,3,intrinsicMatInOut).convertTo( camMat_est, CV_64F );
    }

    std::vector<std::vector<cv::Point3f>> test3d_list( 1 );
    test3d_list.front().assign( pts3d, pts3d + ptCount );
    
    std::vector<std::vector<cv::Point2f>> test2d_list( 1 );
    test2d_list.front().assign( pts2d, pts2d + ptCount );

    cv::Size2f const imgSize(imgWidth,imgHeight);

    int const flags = 
        (intrinsicMatInOut ? CV_CALIB_USE_INTRINSIC_GUESS : 0) |
        CV_CALIB_ZERO_TANGENT_DIST |
        CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;

    double rep_err = cv::calibrateCamera(test3d_list, test2d_list, imgSize,
        camMat_est, distCoeffs_est, rvecs, tvecs, flags );

    if( translationOut )
    {
        cv::Mat1f tempOut(1,3,translationOut);
        tvecs.front().convertTo( tempOut, CV_32F );
    }

    if( rotationOut )
    {
        cv::Mat1d temp(3,3);
        Rodrigues( rvecs.front(), temp );

        cv::Mat1f tempOut(3,3,rotationOut);
        temp.convertTo( tempOut, CV_32F );
    }

    if( intrinsicMatInOut )
    {
        cv::Mat1f tempOut(3,3,intrinsicMatInOut);
        camMat_est.convertTo( tempOut, CV_32F );
    }

    return rep_err;
}