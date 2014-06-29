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

//---------------------------------------------------------------------------------------
// Random number generator
std::mt19937_64 randEng;

void SeedRandomEngine()
{
    union
    {
        FILETIME ft;
        unsigned __int32 i[4];
    } x;
    GetSystemTimeAsFileTime( &x.ft );
    unsigned long seed = x.i[0] ^ x.i[1] ^ x.i[2] ^ x.i[3];
    randEng.seed( seed );
}

//---------------------------------------------------------------------------------------
void GenIntrinsic( float width, float height, float fov_h, float fov_v, Mat1d & intrinsic )
{
    float const center_x = width / 2;
    float const center_y = height / 2;

    float const focal_x = FocalFromFov( width, fov_h );
    float const focal_y = FocalFromFov( height, fov_v );

    intrinsic.create(3,3);
    intrinsic <<
        focal_x,    0.f,        center_x,
        0.f,        focal_y,    center_y,
        0.f,        0.f,        1.f;
}

//---------------------------------------------------------------------------------------
void GenExtrinsic( Mat1d & rotation, Mat1d & translation, Mat1d & extrinsic )
{
    float const rot_x = 30.f;
    float const rot_y = 13.f;
    float const rot_z = 63.f;

    // Rotation and translation are refs to subblocks of extrinsic
    extrinsic.create( 3, 4 );
    rotation = extrinsic.colRange( 0, 3 );
    translation = extrinsic.col( 3 );

    MakeRotation( rot_x, rot_y, rot_z ).copyTo( rotation );
    translation << 1, 2, 3;
}

//---------------------------------------------------------------------------------------
// Generates a pattern of points in camera space (looking down the Z axis)
void MakeCalibPattern( float fov_h, float fov_v, std::vector<Point3f> & result )
{
    enum {
        I_COUNT = 3, I_MAX = I_COUNT - 1,
        J_COUNT = 3, J_MAX = J_COUNT - 1,
        K_COUNT = 2, K_MAX = K_COUNT - 1,

        TOTAL_COUNT = I_COUNT * J_COUNT * K_COUNT
    };
    
    float const inset = 0.8;

    float const slope_h = tan( fov_h / 2 );
    float const slope_v = tan( fov_v / 2 );
    cout << "slope: " << slope_h << ", " << slope_v << endl;

    result.clear();
    result.reserve( TOTAL_COUNT );

    for( size_t i = 0; i < I_COUNT; ++i )
    {
        for( size_t j = 0; j < J_COUNT; ++j )
        {
            for( size_t k = 0; k < K_COUNT; ++k )
            {
                // Normalized image point, +/- 1
                float const u = normalize(i,I_MAX) * 2 - 1;
                float const v = normalize(j,J_MAX) * 2 - 1;

                float const depth = 2.f + 6.f * normalize(k,K_MAX) ;//+ depthRand(randEng);
                float const y = v * inset * (slope_v * depth); // scale by depth
                float const x = u * inset * (slope_h * depth); // scale by depth
                
                //cout
                //    << u << ", "
                //    << (slope_h * depth) << ", "
                //    << x << ", "
                //    << endl;

                result.push_back( Point3f(x, y, depth) );
            }
        }
    }
}

double RotationError( Mat1d const & r )
{
    //http://stackoverflow.com/questions/6522108/error-between-two-rotations

    Point3d p;
    p.x = r(1,2) - r(2,1);
    p.y = r(2,0) - r(0,2);
    p.z = r(0,1) - r(1,0);

    auto dmag = cv::norm( p );

    return asin (dmag/2);
}

//---------------------------------------------------------------------------------------

int _tmain(int argc, _TCHAR* argv[])
{
    SeedRandomEngine();

    // Image size
    float const width = 1.f;
    float const height = 1.f;

    float const fov_h = ToRadians( 30.f );
    float const fov_v = ToRadians( 18.f );

    Mat1d intrinsic;
    GenIntrinsic( width, height, fov_h, fov_v, intrinsic );

    Mat1d rotation, translation, extrinsic;
    GenExtrinsic( rotation, translation, extrinsic );
    Mat1d const invRotation = rotation.t();

    cout << "intrinsic matrix:" << endl
         << intrinsic << endl;
  
    cout << "Rotation matrix:" << endl
         << rotation << endl;

    cout << "translation vector:" << endl
         << translation << endl;

    cout << "extrinsic matrix:" << endl
         << extrinsic << endl;

    Mat1d projection = intrinsic * extrinsic;
    cout << "projection matrix:" << endl
         << projection << endl;

    
    std::normal_distribution<float> imgPtErrDist(0.f, 0.05f); // portion of normalized screen
    std::normal_distribution<float> depthRand(0.0f, 1.f);


    //WaitForEnter();

    vector<Point3f> calibPattern3d;
    MakeCalibPattern( fov_h, fov_v, calibPattern3d );
    
    enum { TEST_MULT = 6 };
    size_t const TOTAL_COUNT = calibPattern3d.size() * TEST_MULT;

    vector< vector<Point3f> > points3d_list(1);
    vector<Point3f> & points3d = points3d_list.front();
    points3d.reserve( TOTAL_COUNT );

    vector< vector<Point2f> > points2d_list(1);
    vector<Point2f> & points2d = points2d_list.front();
    points2d.reserve( TOTAL_COUNT );

    Mat1d worldPtMat;
    Mat1d imgPtHomo;
    for( size_t i = 0; i < calibPattern3d.size(); ++i )
    {
        auto const & calibPt3d = calibPattern3d[i];
        Mat1d const cameraPt = (Mat1d(3,1) << calibPt3d.x, calibPt3d.y, calibPt3d.z);

        //cout << rotationMatrix.t().size() << cameraPt.size() << endl;
        
        worldPtMat = invRotation * (cameraPt - translation);
        imgPtHomo = projection(Rect(0,0,3,3)) * worldPtMat + projection.col(3);
        
        Point3f const worldPt( worldPtMat(0), worldPtMat(1), worldPtMat(2) );
        
        float const w = imgPtHomo(2);
        Point2f const imgPt( imgPtHomo(0) / w, imgPtHomo(1) / w );

        for( auto j = 0; j < TEST_MULT; ++j )
        {
            Point2f const imgPtErr( imgPtErrDist(randEng), imgPtErrDist(randEng) );

            points3d.push_back( worldPt );
            points2d.push_back( imgPt + imgPtErr );
        }

        //cout << i << ": "
        //    << cameraPt << ',' << points2d.back() << ',' << endl;
    }

    Mat camMat_est;
    Mat distCoeffs_est;
    vector<Mat> rvecs(1), tvecs(1);

    try
    {
        cout << endl << "Calibrating...";
        //double rep_err = calibrateCamera(points3d_list, points2d_list, cv::Size2f(width,height), camMat_est, distCoeffs_est, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS );
        double rep_err = solvePnP( points3d, points2d, intrinsic, distCoeffs_est, rvecs.front(), tvecs.front() );
        cout << "Done: " << rep_err << ' ' << (rep_err/TOTAL_COUNT) << endl;

        //cout << endl << "Average Reprojection error: " << rep_err << endl;
        cout << "==================================" << endl;
        
        double tErrMag = 0;
        {
            Mat1d terr = tvecs.front() - translation;
            tErrMag = cv::norm( terr );
            std::for_each( terr.begin(), terr.end(), ReducePrecision() );
            cout << "Translation diff: " << terr << endl
                 << " = " << tvecs.front() << " - " << translation << endl << endl;
        }

        Mat1d outRotation(3,3);
        Rodrigues( rvecs.front(), outRotation );

        cout << rvecs.front() << endl << endl;
        cout << outRotation << endl << endl;

        Mat1d rotDelta = (invRotation * outRotation);
        std::for_each( rotDelta.begin(), rotDelta.end(), ReducePrecision() );
        double rotError = RotationError( rotDelta );
        cout << "Rotation diff:" << endl
             << rotDelta << endl
             << endl;

        cout << "Rotation Error:   " << ToDegrees( rotError ) << " degrees" << endl
             << "Translation error: " << (tErrMag * 12) << " inches" << endl
             << endl;

        cout << camMat_est << endl << endl;

        {
            Mat projectedPts;
            projectPoints( Mat(points3d), rvecs.front(), tvecs.front(), intrinsic, distCoeffs_est, projectedPts );
            Mat temp(points2d);
            double err = norm( temp, projectedPts, CV_L2);              // difference

            //Mat diff = temp - projectedPts;
            //Mat mag;
            //cv::magnitude( diff.col(0), diff.col(1), mag );

            int n = (int)points3d.size();
            cout << "Reprojection Error: " << err << " / " << n << " = " << (err/n) << " => " << (1280*err/n) << " pixels" << endl;
        }
    }
    catch(cv::Exception const & e )
    {
        cerr << e.what() << endl;
    }

    WaitForEnter();

	return 0;
}

