#include "stdafx.h"

#define _USE_MATH_DEFINES

#include <opencv2\opencv.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <vector>

//! @brief Converts field-of-view into focal length, given an image size
template<typename T>
T FocalFromFov(T const & len, T const & fov)
{
    T const halfLen = len / 2;
    T const halfAngle = fov / 2;
    T const slope = tan( halfAngle );
    return static_cast<T>( halfLen / slope );
}

//---------------------------------------------------------------------------------------
//! @brief Generate a basic Intrinsic camera matrix for a given image size and field-of-view
//! @param width Image width in pixels
//! @param height Image height in pixels
//! @param fov_h The horizontal field-of-view
//! @param fov_v The vertical field-of-view
//! @param intrinsic Output param for the result
void GenIntrinsicMat( float width, float height, float fov_h, float fov_v, cv::Mat1f & intrinsic )
{
    float const center_x = width / 2;
    float const center_y = height / 2;

    float const focal_x = FocalFromFov( width, fov_h );
    float const focal_y = FocalFromFov( height, fov_v );

    intrinsic <<
        focal_x,    0.f,        center_x,
        0.f,        focal_y,    center_y,
        0.f,        0.f,        1.f;
}


//=======================================================================================
//=======================================================================================
// EXPORTED FUNCTIONS
extern "C" {

//---------------------------------------------------------------------------------------
//! @brief Generate a basic Intrinsic camera matrix for a given image size and field-of-view
//! @param width Image width in pixels
//! @param height Image height in pixels
//! @param fov_h The horizontal field-of-view
//! @param fov_v The vertical field-of-view
//! @param intrinsic Len 9 array for a 3x3 row-major matrix
__declspec(dllexport)
void GenIntrinsic( float width, float height, float fov_h, float fov_v, float * intrinsic )
{
    cv::Mat1f temp(3,3,intrinsic);
    GenIntrinsicMat( width, height, fov_h, fov_v, temp );
}

//! @brief Wraps OpenCV camera calibration of 2D-3D points correspondences
//! @param pts3d An array of 3D points. Each point pairs with a point in pts2d, by index.
//! @param pts2d An array of 2D points. Each point pairs with a point in pts3d, by index.
//! @param ptCount The number of points in pts3d (and pts2d)
//! @param imgWidth The width of the 2D image in pixels
//! @param imgHeight The height of the 2D image in pixels
//! @param intrinsicMatInOut In/Out parameter for the Intrinsic Camera matrix (len 9 array, 3x3, row-major)
//! @param rotationOut Out parameter for the Camera rotation matrix (len 9 array, 3x3, row-major)
//! @param translationOut Out parameter for the Camera translation vector matrix (len 3)
//! @returns The reprojection error in pixels
//! @note The Extrinsic Camera matrix is made by composing the rotation matrix (3x3) and the translation vector (len 3)
//!     into a 3x4 matrix. The Camera Projection matrix is then:
//!     projection = intrinsic * extrinsic;
__declspec(dllexport)
double CalibCameraWrapperEx(
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

//! @brief Wraps OpenCV camera calibration of 2D-3D points correspondences
//! @param pts3d An array of 3D points. Each point pairs with a point in pts2d, by index.
//! @param pts2d An array of 2D points. Each point pairs with a point in pts3d, by index.
//! @param ptCount The number of points in pts3d (and pts2d)
//! @param imgWidth The width of the 2D image in pixels
//! @param imgHeight The height of the 2D image in pixels
//! @param fov_h Horizontal field-of-view, in radians
//! @param fov_v Vertical field-of-view, in radians
//! @param intrinsicOut Out parameter for the Intrinsic Camera matrix (len 9 array, 3x3, row-major)
//! @param rotationOut Out parameter for the Camera rotation matrix (len 9 array, 3x3, row-major)
//! @param translationOut Out parameter for the Camera translation vector matrix (len 3)
//! @returns The reprojection error in pixels
//! @note The Extrinsic Camera matrix is made by composing the rotation matrix (3x3) and the translation vector (len 3)
//!     into a 3x4 matrix. The Camera Projection matrix is then:
//!     projection = intrinsic * extrinsic;
__declspec(dllexport)
double CalibCameraWrapper(
    cv::Point3f const * pts3d, cv::Point2f const * pts2d, unsigned int ptCount,
    unsigned int imgWidth, unsigned int imgHeight,
    float fov_h, float fov_v,
    float * intrinsicMatOut,
    float * rotationOut, float * translationOut )
{
    float intrinTemp[9];
    GenIntrinsic( imgWidth, imgHeight, fov_h, fov_v, intrinTemp );
    auto result = CalibCameraWrapperEx( pts3d, pts2d, ptCount, imgWidth, imgHeight, intrinTemp, rotationOut, translationOut );
    if( intrinsicMatOut )
        memcpy( intrinsicMatOut, intrinTemp, sizeof( intrinTemp ) );
    return result;
}

} // extern "C"