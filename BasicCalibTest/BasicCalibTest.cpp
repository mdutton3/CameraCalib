// BasicCalibTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2\calib3d\calib3d.hpp>

#include <random>

using namespace std;
using namespace cv;

template<typename T>
T ToRadians(T const & degrees)
{
    return static_cast<T>( degrees * M_PI / 180 );
}

template<typename T>
T ToDegrees(T const & radians)
{
    return static_cast<T>( radians * 180 / M_PI );
}

template<typename T>
T FocalFromFov(T const & len, T const & fov)
{
    T const halfLen = len / 2;
    T const halfAngle = fov / 2;
    T const slope = tan( halfAngle );
    return static_cast<T>( halfLen / slope );
}

template<typename T>
T FovFromFocal(T const & len, T const & focal)
{
    T const halfLen = len / 2;
    T const slope = halfLen / focal;
    T const halfAngle = atan( slope );
    return static_cast<T>( 2 * halfAngle );
}

static void WaitForEnter()
{
    cout << "Press enter" << endl;
    cin.putback('x');
    string dummy;
    cin >> dummy;
}

Mat1f const MakeRotation( float rx, float ry, float rz )
{
    float const sinx = sin( rx );
    float const cosx = cos( rx );
    float const siny = sin( ry );
    float const cosy = cos( ry );
    float const sinz = sin( rz );
    float const cosz = cos( rz );

    Mat1f Rx(3,3);
    Rx <<   1.f,    0.f,    0.f,
            0.f,    cosx,   -sinx,
            0.f,    sinx,   cosx;

    Mat1f Ry(3,3);
    Ry <<   cosy,   0.f,   siny,
            0.f,    1.f,   0.f,
            -siny,  0.f,   cosy;

    Mat1f Rz(3,3);
    Rz <<   cosz,   -sinz,  0.f,
            sinz,   cosz,   0.f,
            0.f,    0.f,    1.f;

    return Rz*Ry*Rx;
}

template<typename T, typename U>
float normalize( T const val, U const maxVal )
{
    return maxVal > 0 ? static_cast<float>(val) / maxVal : 1.f;
}

struct ReducePrecision
{
    template<typename T>
    T operator()( T & val ) const
    {
        auto temp = val * 10000;
        temp = (temp > 0.f) ? floor(temp + 0.5f) : ceil(temp - 0.5f);
        val = std::floor( temp ) / 10000;
        return val;
    }
};

int _tmain(int argc, _TCHAR* argv[])
{
    float const width = 1.f;
    float const height = 1.f;

    float const center_x = width / 2;
    float const center_y = height / 2;

    float const fov_h = ToRadians( 30.f );
    float const fov_v = ToRadians( 18.f );

    float const focal_x = FocalFromFov( width, fov_h );
    float const focal_y = FocalFromFov( height, fov_v );

    Mat1d cameraMatrix( 3, 3 );
    cameraMatrix <<
        focal_x,    0.f,        center_x,
        0.f,        focal_y,    center_y,
        0.f,        0.f,        1.f;

    cout << "Camera matrix:" << endl
         << cameraMatrix << endl;
    
    float const rot_x = 30.f;
    float const rot_y = 13.f;
    float const rot_z = 63.f;

    Mat1d rotationMatrix = MakeRotation( rot_x, rot_y, rot_z );
    cout << "Rotation matrix:" << endl
         << rotationMatrix << endl;

    Mat1d translation = (Mat1d(3,1) << 1, 2, 3);
    cout << "translation vector:" << endl
         << translation << endl;

    Mat1d extrinsic(3,4);
    rotationMatrix.copyTo( extrinsic.colRange(0,3) );
    translation.copyTo( extrinsic.col(3) );
    cout << "extrinsic matrix:" << endl
         << extrinsic << endl;

    Mat1d projection = cameraMatrix * extrinsic;
    cout << "projection matrix:" << endl
         << projection << endl;

    std::mt19937_64 randEng;
    std::normal_distribution<float> imgPtErr(0.f, 0.05f); // portion of normalized screen
    std::normal_distribution<float> depthRand(0.0f, 1.f);

    float const slope_h = tan( fov_h / 2 );
    float const slope_v = tan( fov_v / 2 );
    Mat1d const rot_inv = rotationMatrix.t();

    cout << "slope: " << slope_h << ", " << slope_v << endl;
    //WaitForEnter();

    enum {
        I_COUNT = 4, I_MAX = I_COUNT - 1,
        J_COUNT = 4, J_MAX = J_COUNT - 1,
        K_COUNT = 2, K_MAX = K_COUNT - 1,

        TOTAL_COUNT = I_COUNT * J_COUNT * K_COUNT
    };
    
    vector< vector<Point3f> > points3d_list(1);
    vector<Point3f> & points3d = points3d_list.front();
    points3d.reserve( TOTAL_COUNT );

    vector< vector<Point2f> > points2d_list(1);
    vector<Point2f> & points2d = points2d_list.front();
    points2d.reserve( TOTAL_COUNT );

    for( size_t i = 0; i < I_COUNT; ++i )
    {
        for( size_t j = 0; j < J_COUNT; ++j )
        {
            for( size_t k = 0; k < K_COUNT; ++k )
            {
                float const z = 2.f + 6.f * normalize(k,K_MAX) ;//+ depthRand(randEng);
                float const y = (normalize(j,J_MAX) - 0.5f) * 2 * (slope_v * z);
                float const x = (normalize(i,I_MAX) - 0.5f) * 2 * (slope_h * z);
                cout
                    << normalize(i,I_MAX) << ", "
                    << (normalize(i,I_MAX) - 0.5f) << ", "
                    << (slope_h * z) << ", "
                    << x << ", "
                    << endl;

                Mat1d const cameraPt = (Mat1d(3,1) << x,y,z);
                //cout << rotationMatrix.t().size() << cameraPt.size() << endl;
                Mat1d worldPt( rot_inv * (cameraPt - translation) );
                points3d.push_back( Point3f( worldPt(0), worldPt(1), worldPt(2) ) );

                //cout << projection.size() << worldPtHomo.size() << endl;
                Mat1d const imgPt = projection(Rect(0,0,3,3)) * worldPt + projection.col(3);
                float u = (imgPt(0) / imgPt(2)) + imgPtErr(randEng);
                float v = (imgPt(1) / imgPt(2)) + imgPtErr(randEng);
                points2d.push_back( Point2f( u,v ) );

                cout << i << ',' << j << ',' << k << ": "
                    << cameraPt << ',' << points2d.back() << ',' << endl;
            }
        }
    }

    Mat camMat_est;
    Mat distCoeffs_est;
    vector<Mat> rvecs(1), tvecs(1);

    try
    {
        cout << endl << "Calibrating...";
        //double rep_err = calibrateCamera(points3d_list, points2d_list, cv::Size2f(width,height), camMat_est, distCoeffs_est, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS );
        double rep_err = solvePnP( points3d, points2d, cameraMatrix, distCoeffs_est, rvecs.front(), tvecs.front() );
        cout << "Done: " << rep_err << ' ' << (rep_err/TOTAL_COUNT) << endl;

        //cout << endl << "Average Reprojection error: " << rep_err << endl;
        cout << "==================================" << endl;
        
        {
            Mat1d terr = tvecs.front() - translation;
            std::for_each( terr.begin(), terr.end(), ReducePrecision() );
            cout << terr << " = " << tvecs.front() << " - " << translation << endl << endl;
        }

        Mat1d outRotation(3,3);
        Rodrigues( rvecs.front(), outRotation );

        cout << rvecs.front() << endl << endl;
        cout << outRotation << endl << endl;

        Mat1d rotError = (rot_inv * outRotation);
        std::for_each( rotError.begin(), rotError.end(), ReducePrecision() );
        cout << rotError << endl << endl;

        cout << camMat_est << endl << endl;

        {
            Mat projectedPts;
            projectPoints( Mat(points3d), rvecs.front(), tvecs.front(), cameraMatrix, distCoeffs_est, projectedPts );
            Mat temp(points2d);
            double err = norm( temp, projectedPts, CV_L2);              // difference

            //Mat diff = temp - projectedPts;
            //Mat mag;
            //cv::magnitude( diff.col(0), diff.col(1), mag );

            int n = (int)points3d.size();
            cout << "Error: " << err << " / " << n << " = " << (err/n) << " => " << (1024*err/n) << endl;
        }
    }
    catch(cv::Exception const & e )
    {
        cerr << e.what() << endl;
    }

    WaitForEnter();

	return 0;
}

