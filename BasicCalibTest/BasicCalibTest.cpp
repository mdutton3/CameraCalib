// BasicCalibTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2\calib3d\calib3d.hpp>

#include <random>
#include <fstream>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

#include <boost/asio.hpp>
#include <boost/asio/io_service.hpp>

#include <boost/bind.hpp>

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

// Image size
float const width = 1280;
float const height = 720.f;

float const fov_h = ToRadians( 30.f );
float const fov_v = ToRadians( 18.f );

//Mat s_img( height, width, CV_8UC3, CV_RGB(0,0,0) );
void ClearImage()
{
    //s_img = CV_RGB(0,0,0);
}

//---------------------------------------------------------------------------------------
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
void MakeCalibPattern( int I_COUNT, int J_COUNT, int K_COUNT, float fov_h, float fov_v, std::vector<Point3f> & result3d )
{
    auto const I_MAX = I_COUNT - 1;
    auto const J_MAX = J_COUNT - 1;
    auto const K_MAX = K_COUNT - 1;
    auto const TOTAL_COUNT = I_COUNT * J_COUNT * K_COUNT;
    
    float const inset = 0.8f;

    float const slope_h = tan( fov_h / 2 );
    float const slope_v = tan( fov_v / 2 );
    //cout << "slope: " << slope_h << ", " << slope_v << endl;

    result3d.clear();
    result3d.reserve( TOTAL_COUNT );

    for( int i = 0; i < I_COUNT; ++i )
    {
        for( int j = 0; j < J_COUNT; ++j )
        {
            for( int k = 0; k < K_COUNT; ++k )
            {
                // Normalized image point, (+/- inset)
                float const u = inset * (normalize(i,I_MAX) * 2 - 1);
                float const v = inset * (normalize(j,J_MAX) * 2 - 1);

                float const depth = 2.f + 6.f * normalize(k,K_MAX) ;//+ depthRand(randEng);
                float const y = v * (slope_v * depth); // scale by depth
                float const x = u * (slope_h * depth); // scale by depth
                
                //cout
                //    << u << ", "
                //    << (slope_h * depth) << ", "
                //    << x << ", "
                //    << endl;

                result3d.push_back( Point3f( x, y, depth ) );
            }
        }
    }
}

//---------------------------------------------------------------------------------------
void CameraToWorldCoord( Mat1d const & extrinsic, std::vector<Point3f> & points )
{
    Mat1d invRotation = extrinsic.colRange(0,3).t();
    Mat1d translation = extrinsic.col(3);

    // Convert camera coord to world coord
    Mat1d ptMat(3,1);
    Mat1d worldPtMat(3,1);
    for( size_t i = 0; i < points.size(); ++i )
    {
        auto & pt = points[i];
        ptMat << pt.x, pt.y, pt.z;
        worldPtMat = invRotation * (ptMat - translation);
        pt = Point3f( worldPtMat(0), worldPtMat(1), worldPtMat(2) );
    }
}

//---------------------------------------------------------------------------------------
void ProjectCalibPattern( Mat1d const & projection, std::vector<Point3f> const & points3d, std::vector<Point2f> & points2d )
{
    points2d.resize( points3d.size() );

    Mat1d pt3d(3,1);
    Mat1d pt2dHomo(3,1);
    for( size_t i = 0; i < points3d.size(); ++i )
    {
        pt3d << points3d[i].x, points3d[i].y, points3d[i].z;

        pt2dHomo = (projection.colRange(0,3) * pt3d) + projection.col(3);
        double w = pt2dHomo(2);
        points2d[i].x = pt2dHomo(0) / w;
        points2d[i].y = pt2dHomo(1) / w;
    }
}

//---------------------------------------------------------------------------------------
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
void GenTestPoints( std::vector<Point3f> const & truth3d, std::vector<Point2f> const & truth2d,
                    double imgPtErrStdDev,
                    std::vector<Point3f> & test3d, std::vector<Point2f> & test2d )
{
    size_t const count = truth3d.size();
    test2d.reserve( test2d.size() + count );
    test3d.reserve( test3d.size() + count );

    std::normal_distribution<float> imgPtErrDist(0.f, imgPtErrStdDev); // portion of normalized screen
    std::uniform_real_distribution<double> angleDist( 0.0, 2 * M_PI );

    for( size_t i = 0; i < count; ++i )
    {
        auto const & calibPt3d = truth3d[i];
        auto const & calibPt2d = truth2d[i];

        //cv::circle( s_img, calibPt2d, imgPtErrStdDev*2, CV_RGB(240,0,0), 1 );
        //cv::circle( s_img, calibPt2d, imgPtErrStdDev  , CV_RGB(255,0,0), 1 );
        //cv::circle( s_img, calibPt2d, 1  , CV_RGB(255,0,0), 1 );
        
        double errDist = imgPtErrDist(randEng);
        double errAngle = angleDist( randEng );
        Point2f const imgPtErr( sin(errAngle) * errDist, cos(errAngle) * errDist );

        auto point2d = calibPt2d + imgPtErr;

        test3d.push_back( calibPt3d );
        test2d.push_back( point2d );
        
        //cv::circle( s_img, point2d, 5, CV_RGB(0,255,0), 1 );
    }
}

void Eval( Mat1d const & intrinsic, Mat1d const & extrinsic,
           std::vector<Point3f> const & truth3d, std::vector<Point2f> const & truth2d,
           std::vector<Point3f> & test3d, std::vector<Point2f> & test2d,
           double & rotError, double & transError, double & projError )
{
    Mat distCoeffs_est;
    vector<Mat> rvecs(1), tvecs(1);

    try
    {
        //cout << endl << "Calibrating...";

        bool const bFullCalib = true;
        if( bFullCalib )
        {
            Mat camMat_est = intrinsic.clone();
            vector<vector<Point3f>> test3d_list( 1, test3d );
            vector<vector<Point2f>> test2d_list( 1, test2d );

            double rep_err = calibrateCamera(test3d_list, test2d_list, cv::Size2f(width,height),
                camMat_est, distCoeffs_est, rvecs, tvecs,
                CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_ZERO_TANGENT_DIST |
                CV_CALIB_FIX_K2 | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6 );
            //cout << "rep_err: " << rep_err << endl;
            //cout << "camMat_est: " << endl << camMat_est << endl;
            //cout << "distCoeffs_est: " << distCoeffs_est << endl;
            //cout << endl;
        }
        else
        {
            solvePnP( test3d, test2d, intrinsic, distCoeffs_est, rvecs.front(), tvecs.front() );
        }        

        Mat1d const translation = extrinsic.col(3);
        transError = cv::norm( tvecs.front() - translation );
        transError *= 12;

        Mat1d outRotation(3,3);
        Rodrigues( rvecs.front(), outRotation );

        Mat1d rotDelta = (extrinsic.colRange(0,3).t() * outRotation);
        rotError = RotationError( rotDelta );
        rotError = ToDegrees( rotError );

        //cout << rvecs.front() << endl << endl;
        //cout << outRotation << endl << endl;        

        //std::for_each( rotDelta.begin(), rotDelta.end(), ReducePrecision() );
        //cout << "Rotation diff:" << endl
        //     << rotDelta << endl
        //     << endl;

        //cout << "Rotation Error:   " << ToDegrees( rotError ) << " degrees" << endl
        //     << "Translation error: " << (transError * 12) << " inches" << endl;

        {
            Mat projectedPts;
            projectPoints( Mat(truth3d), rvecs.front(), tvecs.front(), intrinsic, distCoeffs_est, projectedPts );
            
            double err = cv::norm( projectedPts, Mat(truth2d), CV_L2 );

            int n = (int)truth3d.size();
            projError = (err/n);
            //cout << "Reprojection Error: " << err << " / " << n << " = " << projError << " pixels" << endl;

            //for( auto i = 0; i < projectedPts.rows; ++i )
            //{
            //    Point2d p( projectedPts.at<Point2f>(i) );
            //    cv::circle( s_img, p, 5, CV_RGB(0,0,255), 1 );
            //}
        }
    }
    catch(cv::Exception const & e )
    {
        cerr << e.what() << endl;
    }

    //cout << endl;
}

//---------------------------------------------------------------------------------------
void GetStats( double const * data, size_t count, double & min, double & max, double &mean, double &var, double&bound )
{
    std::vector<double> temp( data, data + count );

    min = max = data[0];
    
    std::sort( temp.begin(), temp.end(), std::less<double>() ); // sort for numeric stability
    double total = 0;
    for( size_t i = 0; i < count; ++i )
        total += temp[i];

    mean = total / count;

    double varAccum = 0;
    for( size_t i = 0; i < count; ++i )
    {
        temp[i] -= mean;
        temp[i] = pow(temp[i],2);
    }

    std::sort( temp.begin(), temp.end(), std::less<double>() ); // sort for numeric stability
    for( size_t i = 0; i < count; ++i )
        varAccum += temp[i];

    if( count > 1 )
        var = varAccum / (count - 1);
    else
        var = 0;

    double stddev = sqrt( var );
    bound = mean + (2 * stddev);
}

//---------------------------------------------------------------------------------------
boost::mutex s_logMutex;
//std::ofstream cycleLog;
std::ofstream batchLog;

void DoBatch( Mat1d const & intrinsic_, Mat1d const &extrinsic_, Mat1d const & projection_, int const i, int const j, int const k )
{
    Mat1d const intrinsic( intrinsic_.clone() );
    Mat1d const extrinsic( extrinsic_.clone() );
    Mat1d const projection( projection_.clone() );

    vector<Point3f> calibPattern3d_;
    vector<Point2f> calibPattern2d_;
    vector<Point3f> test3d;
    vector<Point2f> test2d;

    MakeCalibPattern( i,j,k, fov_h, fov_v, calibPattern3d_ );
    CameraToWorldCoord( extrinsic, calibPattern3d_ );
    
    ProjectCalibPattern( projection, calibPattern3d_, calibPattern2d_ );

    vector<Point3f> const & calibPattern3d( calibPattern3d_ );
    vector<Point2f> const & calibPattern2d( calibPattern2d_ );

    for( int SAMPLE_COUNT = 1; SAMPLE_COUNT <= 20; ++SAMPLE_COUNT )
    for( int errStdDevPixel = 1; errStdDevPixel <= 25; errStdDevPixel += 2 )
    {
        size_t const count = i * j * k * SAMPLE_COUNT;
        if( count > 300 )
            continue;

        enum { NUM_ITER = 200 };
        double rotError[NUM_ITER], transError[NUM_ITER], projError[NUM_ITER];
        for( int ITER = 0; ITER < NUM_ITER; ++ITER )
        {
            ClearImage();
            test3d.clear();
            test2d.clear();
    
            for( int iMult = 0; iMult < SAMPLE_COUNT; ++iMult )
            {
                GenTestPoints( calibPattern3d, calibPattern2d, errStdDevPixel, test3d, test2d );
            }

            //cv::imwrite( "calibpts.png", s_img );
            //cv::imshow( "2d pts", s_img );
            //cv::waitKey();
            //cv::destroyWindow("2d pts");

            Eval( intrinsic, extrinsic,
                calibPattern3d, calibPattern2d,
                test3d, test2d,
                rotError[ITER], transError[ITER], projError[ITER] );

            //cycleLog
            //    << i << ", "
            //    << j << ", "
            //    << k << ", "
            //    << SAMPLE_COUNT << ", "
            //    << errStdDevPixel << ", "
            //    << test3d.size() << ", "
            //    << "Data" << ", "
            //    << rotError[ITER] << ", "
            //    << transError[ITER] << ", "
            //    << projError[ITER] << endl;
            //cv::imwrite( "final.png", s_img );
            //cv::imshow( "2d pts", s_img );
            //cv::waitKey();
            //cv::destroyWindow("2d pts");
        }

        {
            boost::lock_guard<boost::mutex> lockGuard( s_logMutex );
            cout
                << i << ", "
                << j << ", "
                << k << ", "
                << SAMPLE_COUNT << ", "
                << errStdDevPixel << ", "
                << test3d.size() << ", "
                << "Done" << std::endl;

            batchLog
                << i << ", "
                << j << ", "
                << k << ", "
                << SAMPLE_COUNT << ", "
                << errStdDevPixel << ", "
                << test3d.size() << ", "
                << "Data" << ", ";

            double minVal, maxVal, meanVal, var, bound;
            GetStats( rotError, NUM_ITER, minVal, maxVal, meanVal, var, bound );
            batchLog << "rotError" << ","
                << minVal << ","
                << maxVal << ","
                << meanVal << ","
                << var << ","
                << bound << ",";

            GetStats( transError, NUM_ITER, minVal, maxVal, meanVal, var,bound );
            batchLog << "transError" << ","
                << minVal << ","
                << maxVal << ","
                << meanVal << ","
                << var << ","
                << bound << ",";

            GetStats( projError, NUM_ITER, minVal, maxVal, meanVal, var, bound );
            batchLog << "projError" << ","
                << minVal << ","
                << maxVal << ","
                << meanVal << ","
                << var << ","
                << bound << ",";

            batchLog << endl;
        }
    }
}

int _tmain(int argc, _TCHAR* argv[])
{
    SeedRandomEngine();

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



    //cycleLog.open( "cycle.log.csv" );
    batchLog.open( "batch.log.csv" );
    {
        //cycleLog << "i, j, k, nSample, errStdDev, nPt, xx, errRot, errTrans, errProj" << endl;
        batchLog << "i, j, k, nSample, errStdDev, nPt, xx"
            << ", _errRot_, errRot_min, errRot_max, errRot_mean, errRot_var, errRot_bound"
            << ", _errTrans_, errTrans_min, errTrans_max, errTrans_mean, errTrans_var, errTrans_bound"
            << ", _errProj_, errProj_min, errProj_max, errProj_mean, errProj_var, errProj_bound"
            << endl;
    }

    boost::thread_group threadPool;
    boost::asio::io_service service;
    std::auto_ptr<boost::asio::io_service::work> work( new boost::asio::io_service::work(service) );

    for( int i = 0; i < 8; ++i )
    {
        threadPool.create_thread( 
            boost::bind( &boost::asio::io_service::run, &service ) );
    }

    for( int i = 2; i <= 10; ++i )
    for( int j = 2; j <= 10; ++j )
    for( int k = 1; k <= 10; ++k )
    {
        service.post( boost::bind( &DoBatch, intrinsic, extrinsic, projection, i, j, k ) );
    }

    cout << "Waiting for completion..." << endl;
    work.reset();
    threadPool.join_all( );
    service.stop( );

    return 0;
}
