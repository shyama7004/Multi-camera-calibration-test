### Directory structure :

```py

ARUCO-3.1.12/
│── 3rdparty/                  # Third-party dependencies (if any)
│── build/                      # Build directory (usually generated after compilation)
│── cmake/                      # CMake configuration files
│── src/                        # Source code
│   ├── dcf/
│   ├── aruco_cversioning.h
│   ├── aruco_export.h
│   ├── aruco.h
│   ├── cameraparameters.cpp
│   ├── cameraparameters.h
│   ├── CMakeLists.txt
│   ├── cvdrawingutils.cpp
│   ├── cvdrawingutils.h
│   ├── debug.cpp
│   ├── debug.h
│   ├── dictionary_based.cpp
│   ├── dictionary_based.h
│   ├── dictionary.cpp
│   ├── dictionary.h
│   ├── fractaldetector.cpp
│   ├── fractaldetector.h
│   ├── ippe.cpp
│   ├── ippe.h
│   ├── levmarq.h
│   ├── marker.cpp
│   ├── marker.h
│   ├── markerdetector_impl.cpp
│   ├── markerdetector_impl.h
│   ├── markerdetector.cpp
│   ├── markerdetector.h
│   ├── markerlabeler.cpp
│   ├── markerlabeler.h
│   ├── markermap.cpp
│   ├── markermap.h
│   ├── picoflann.h
│   ├── posetracker.cpp
│   ├── posetracker.h
│   ├── timers.h
│
│── utils/                      # Utility functions
│   ├── utils_calibration/      # Calibration-related utilities
│   ├── utils_dcf/              # DCF-related utilities
│   ├── utils_fractal/          # Fractal-related utilities
│   │   ├── fractal_create.cpp
│   │   ├── fractal_pix2meters.cpp
│   │   ├── fractal_print_marker.cpp
│   │   ├── fractal_tracker.cpp
│   ├── utils_gl/               # OpenGL utilities
│   ├── utils_markermap/        # Marker mapping utilities
│
│── CMakeLists.txt              # CMake configuration
│── ChangeLog                   # Changelog file
│── LICENSE                     # License file
│── License-gpl.txt              # GPL License information
│── README                      # Readme file with documentation


```

1. src/fractallabelers/fractallabeler.cpp

```cpp

#include "fractallabeler.h"

#include "../aruco_cvversioning.h"
namespace aruco
{

    void FractalMarkerLabeler::setConfiguration(const FractalMarkerSet& fractMarkerSet) {
        _fractalMarkerSet = fractMarkerSet;
    }

    bool FractalMarkerLabeler::detect(const cv::Mat& in, int& marker_id, int& nRotations, std::string &additionalInfo)
    {
        assert(in.rows == in.cols);
        cv::Mat grey;
        if (in.type() == CV_8UC1)
            grey = in;
        else
            cv::cvtColor(in, grey, CV_BGR2GRAY);
        // threshold image
        cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::map<uint32_t,std::vector<cv::Mat> > nbits_innerCodes;

        for(auto bitsids:_fractalMarkerSet.nbits_fractalMarkerIDs){

            int nbits = bitsids.first;
            std::vector<cv::Mat> innerCodes;
            getInnerCode(grey, nbits, innerCodes);

            if (innerCodes.size()>0){
                if (sum(innerCodes[0])[0]!=0){
                    nbits_innerCodes[nbits]=innerCodes;
                }
            }
        }

        if ( nbits_innerCodes.size()==0)return false;

        //check if any dictionary recognizes it
        for(auto bit_innerCodes:nbits_innerCodes){

            uint32_t nb = bit_innerCodes.first;
            auto innerCodes = bit_innerCodes.second;

            for (int i = 0; i < 4; i++)
            {
                if (_fractalMarkerSet.isFractalMarker(innerCodes[i], nb, marker_id))
                {
                    // is in the set?
                    nRotations = i;  // how many rotations are and its id
                    return true;  // bye bye
                }
            }
        }
        return false;
    }


    std::string FractalMarkerLabeler::getName() const
    {
        return "fractal";;
    }

    bool FractalMarkerLabeler::getInnerCode(const cv::Mat& thres_img, int total_nbits, std::vector<cv::Mat>& innerCodes)
    {
        int bits_noborder = static_cast<int>(std::sqrt(total_nbits));
        int bits_withborder = bits_noborder + 2;
        // Markers  are divided in (bits_a+2)x(bits_a+2) regions, of which the inner bits_axbits_a belongs to marker
        // info
        // the external border shoould be entirely black
        cv::Mat nonZeros(bits_withborder,bits_withborder,CV_32SC1);
        cv::Mat nValues(bits_withborder,bits_withborder,CV_32SC1);
        nonZeros.setTo(cv::Scalar::all(0));
        nValues.setTo(cv::Scalar::all(0));
        for (int y = 0; y <  thres_img.rows; y++)
        {
            const uchar *ptr=thres_img.ptr<uchar>(y);
            int my=   float(bits_withborder)*float(y)/ float(thres_img.rows);
            for (int x = 0; x < thres_img.cols; x++)
            {
                int mx=   float(bits_withborder)*float(x)/ float(thres_img.cols);
                if( ptr[x]>125)
                    nonZeros.at<int>(my,mx)++;
                nValues.at<int>(my,mx)++;
            }
        }
        cv::Mat binaryCode(bits_withborder,bits_withborder,CV_8UC1);
        //now, make the theshold
        for(int y=0;y<bits_withborder;y++)
            for(int x=0;x<bits_withborder;x++){
                 if(nonZeros.at<int>(y,x)>nValues.at<int>(y,x)/2)
                    binaryCode.at<uchar>(y,x)=1;
                else
                    binaryCode.at<uchar>(y,x)=0;
            }

        //check if border is completely black
        for (int y = 0; y < bits_withborder; y++)
       {
           int inc = bits_withborder - 1;
           if (y == 0 || y == bits_withborder - 1)
               inc = 1;  // for first and last row, check the whole border
           for (int x = 0; x < bits_withborder; x += inc)
             if (binaryCode.at<uchar>(y,x)!=0 ) return false;
        }

        //take the inner code
        cv::Mat _bits(bits_noborder,bits_noborder,CV_8UC1);
        for(int y=0;y<bits_noborder;y++)
            for(int x=0;x<bits_noborder;x++)
                _bits.at<uchar>(y,x)=binaryCode.at<uchar>(y+1,x+1);

        // now, get the 64bits ids
        int nr = 0;
        do
        {
            innerCodes.push_back(_bits);
            _bits = rotate(_bits);
            nr++;
        } while (nr < 4);
        return true;
    }

    cv::Mat FractalMarkerLabeler::rotate(const cv::Mat& in)
    {
        cv::Mat out;
        in.copyTo(out);
        for (int i = 0; i < in.rows; i++)
        {
            for (int j = 0; j < in.cols; j++)
            {
                out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
            }
        }
        return out;
    }
}

```

2. src/fractallabelers/fractallabeler.h
```cpp

#include "../markerlabeler.h"
#include "fractalposetracker.h"

namespace aruco
{
    class FractalMarkerLabeler : public MarkerLabeler
    {
    public:
        static cv::Ptr<FractalMarkerLabeler> create(std::string params)
        {
            FractalMarkerSet fractalMarkerSet = FractalMarkerSet::load(params);
            FractalMarkerLabeler* fml=new FractalMarkerLabeler();
            fml->setConfiguration(fractalMarkerSet);
            return fml;
        }

        static cv::Ptr<FractalMarkerLabeler> create(FractalMarkerSet::CONF_TYPES conf)
        {
            FractalMarkerSet fractalMarkerSet = FractalMarkerSet::loadPredefined(conf);
            FractalMarkerLabeler* fml=new FractalMarkerLabeler();
            fml->setConfiguration(fractalMarkerSet);
            return fml;
        }

        void setConfiguration(const FractalMarkerSet& fractMarkerSet);

        static bool isFractalDictionaryFile(const std::string &path);

        virtual ~FractalMarkerLabeler()
        {
        }

        bool load(const std::string &path);

        // returns the configuration name
        std::string getName() const;

        // main virtual class to o detection
        bool detect(const cv::Mat& in, int& marker_id, int& nRotations,std::string &additionalInfo);

        int getNSubdivisions()const{return (sqrt(_fractalMarkerSet.nBits())+2);}

        FractalMarkerSet _fractalMarkerSet;

    private:
        bool getInnerCode(const cv::Mat& thres_img, int total_nbits, std::vector<cv::Mat>& ids);
        cv::Mat rotate(const cv::Mat& in);
    };
}

```

3.src/fractallabelers/fractalmarker.cpp

```cpp

#include "fractalmarker.h"

#include <bitset>


namespace aruco
{
    FractalMarker::FractalMarker()
    {

    }

    FractalMarker::FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers)
    {
        this->id = id;
        this->_M = m;
        for(auto p:corners)
            push_back(p);

        _submarkers = id_submarkers;
        _mask = cv::Mat::ones(m.size(), CV_8UC1);
    }

    void FractalMarker::addSubFractalMarker(FractalMarker submarker)
    {
        float bitSize = (at(1).x - at(0).x) / int(mat().cols+2);
        float nsubBits = (submarker.at(1).x - submarker.at(0).x) / bitSize;

        int x_min = int(round(submarker[0].x / bitSize + mat().cols/2));
        int x_max = x_min + nsubBits;
        int y_min = int(round(-submarker[0].y / bitSize + mat().cols/2));
        int y_max = y_min + nsubBits;

        for(int y=y_min; y<y_max; y++){
            for(int x=x_min; x<x_max; x++){
                _mask.at<uchar>(y,x)=0;
            }
        }
    }

    std::vector<cv::Point3f> FractalMarker::findInnerCorners()
    {
        int nBitsSquared = int(sqrt(mat().total()));
        float bitSize =  getMarkerSize() / (nBitsSquared+2);

        //Set submarker pixels (=1) and add border
        cv::Mat marker;
        mat().copyTo(marker);
        marker +=  -1 * (mask()-1);
        cv::Mat markerBorder;
        copyMakeBorder(marker, markerBorder, 1,1,1,1,cv::BORDER_CONSTANT,0);

        //Get inner corners
        std::vector<cv::Point3f> innerCorners;
        for(int y=0; y< markerBorder.rows-1; y++)
        {
            for(int x=0; x< markerBorder.cols-1; x++)
            {

                if(     ((markerBorder.at<uchar>(y, x) == markerBorder.at<uchar>(y+1, x+1)) &&
                         (markerBorder.at<uchar>(y, x) != markerBorder.at<uchar>(y, x+1) ||
                         markerBorder.at<uchar>(y, x) != markerBorder.at<uchar>(y+1, x)))

                        ||

                        ((markerBorder.at<uchar>(y, x+1) == markerBorder.at<uchar>(y+1, x)) &&
                        (markerBorder.at<uchar>(y, x+1) != markerBorder.at<uchar>(y, x) ||
                         markerBorder.at<uchar>(y, x+1) != markerBorder.at<uchar>(y+1, x+1)))
                    )
                    innerCorners.push_back(cv::Point3f(x-nBitsSquared/2.f, -(y-nBitsSquared/2.f), 0) * bitSize);
            }
        }
        return innerCorners;
    }
}

```

4. src/fractallabelers/fractalmarker.h

```cpp

#ifndef FRACTALMARKER_H
#define FRACTALMARKER_H

#include <vector>
#include <bitset>
#include <opencv2/imgproc/imgproc.hpp>
#include "../markermap.h"

namespace aruco
{
    class FractalMarker : public aruco::Marker3DInfo
    {
        public:
            FractalMarker();
            FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers);

            //Add new submarker
            void addSubFractalMarker(FractalMarker submarker);

            //Find inner corners
            std::vector<cv::Point3f> findInnerCorners();

            //Marker MAT
            const cv::Mat mat() const
            {
                return _M;
            }

            //Marker mask (mask applied to submarkers)
            const cv::Mat mask() const
            {
                return _mask;
            }

            //Total number of bits
            int nBits()
            {
                return _M.total();
            }

            //Submarkers ids
            std::vector<int> subMarkers()
            {
                return _submarkers;
            }

            //Get inner corners
            std::vector<cv::Point3f> getInnerCorners()
            {
                if(innerCorners.empty())
                    innerCorners = findInnerCorners();

                return innerCorners;
            }

        private:
            cv::Mat _M;
            cv::Mat _mask;
            std::vector<int> _submarkers; //id subfractalmarkers
            std::vector<cv::Point3f> innerCorners;
    };
}

#endif // FRACTALMARKER_H

```

5. src/fractallabelers/fractalmarkerset.cpp

```cpp

#include "fractalmarkerset.h"
#include <iostream>
#include <fstream>

namespace aruco
{

FractalMarkerSet FractalMarkerSet::load(std::string info){
    if (isPredefinedConfigurationString(info))
        return loadPredefined(info);
    else return readFromFile(info);
}

FractalMarkerSet FractalMarkerSet::loadPredefined(std::string type){
    return loadPredefined(getTypeFromString(type));
}

FractalMarkerSet FractalMarkerSet::loadPredefined(CONF_TYPES type){
    FractalMarkerSet fms;

    switch(type){
        case FRACTAL_2L_6:{
            unsigned char _conf_2L_6[] = {
                0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x24, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0x3e,
                0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0x3e,
                0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0xbe,
                0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0xbe,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00
            };
            unsigned int _conf_2L_6_len = 272;

            std::stringstream stream;
            stream.write((char*) _conf_2L_6, sizeof(unsigned char)*_conf_2L_6_len);
            _fromStream(fms, stream);
        }break;
        case FRACTAL_3L_6:{
            unsigned char _conf_3L_6[] = {
                0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
                0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
                0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
                0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
                0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
                0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
                0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
                0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
                0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
                0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
                0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
                0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            };
            unsigned int _conf_3L_6_len = 480;

            std::stringstream stream;
            stream.write((char*) _conf_3L_6, sizeof(unsigned char)*_conf_3L_6_len);
            _fromStream(fms, stream);
        }break;
        case FRACTAL_4L_6:{
            unsigned char _conf_4L_6[] = {
                0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00,
                0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
                0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
                0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
                0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
                0x00, 0x64, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
                0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
                0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
                0xbe, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
                0xbe, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
                0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00,
                0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
                0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
                0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
                0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
                0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x00, 0x00, 0x00, 0x00, 0x00
            };
            unsigned int _conf_4L_6_len = 713;

            std::stringstream stream;
            stream.write((char*) _conf_4L_6, sizeof(unsigned char)*_conf_4L_6_len);
            _fromStream(fms, stream);
        }break;
        case FRACTAL_5L_6:{
            unsigned char _conf_5L_6[] = {
                0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
                0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00,
                0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
                0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
                0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
                0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
                0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
                0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
                0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
                0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
                0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
                0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
                0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
                0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
                0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
                0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0xd9, 0x89,
                0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
                0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
                0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
                0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
                0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00,
                0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
                0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
                0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
                0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
                0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01,
                0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01,
                0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
            };
            unsigned int _conf_5L_6_len = 898;

            std::stringstream stream;
            stream.write((char*) _conf_5L_6, sizeof(unsigned char)*_conf_5L_6_len);
            _fromStream(fms, stream);
        }break;
        case CUSTOM:
            throw cv::Exception(-1,"CUSTOM type is only set by loading from file","FractalMarkerSet::loadPredefined","fractalmarkerset.h",-1);
        break;

        default:           throw cv::Exception(9001, "Invalid Dictionary type requested", "Dictionary::loadPredefined", __FILE__, __LINE__);

    }
    return fms;
}

bool FractalMarkerSet::isPredefinedConfigurationString(std::string str)  {
       return getTypeFromString(str)!=CUSTOM;
}

std::string FractalMarkerSet::getTypeString(CONF_TYPES t)  {
        switch(t){
            case FRACTAL_2L_6:return "FRACTAL_2L_6";
            case FRACTAL_3L_6:return "FRACTAL_3L_6";
            case FRACTAL_4L_6:return "FRACTAL_4L_6";
            case FRACTAL_5L_6:return "FRACTAL_5L_6";
            case CUSTOM:return "CUSTOM";
        };
       return "Non valid CONF_TYPE";
    }

FractalMarkerSet::CONF_TYPES FractalMarkerSet::getTypeFromString(std::string str)  {
        if (str=="FRACTAL_2L_6") return FRACTAL_2L_6;
        if (str=="FRACTAL_3L_6") return FRACTAL_3L_6;
        if (str=="FRACTAL_4L_6") return FRACTAL_4L_6;
        if (str=="FRACTAL_5L_6") return FRACTAL_5L_6;
        else  return CUSTOM;
    }

std::vector<std::string> FractalMarkerSet::getConfigurations() {
    return {"FRACTAL_2L_6","FRACTAL_3L_6","FRACTAL_4L_6","FRACTAL_5L_6"};
}

void FractalMarkerSet::_toStream(FractalMarkerSet &configuration, std::ostream &str)
{
    str.write((char*)&configuration.mInfoType,sizeof(mInfoType));
    str.write((char*)&configuration._nmarkers,sizeof(_nmarkers));
    str.write((char*)&configuration._idExternal,sizeof(_idExternal));

    for(auto fractal:configuration.fractalMarkerCollection)
    {
        //ID
        int id = fractal.first;
        str.write((char*)&id,sizeof(id));
        //NBITS
        int nbits = fractal.second.nBits();
        str.write((char*)&nbits,sizeof(nbits));
        //CORNERS
        std::vector<cv::Point3f> corners = fractal.second.points;
        str.write((char*)&corners[0],sizeof(cv::Point3f)*4);
        //MAT
        cv::Mat mat = fractal.second.mat();
        str.write((char*)mat.data, mat.elemSize() * mat.total());
        //SUBMARKERS
        std::vector<int> sub = fractal.second.subMarkers();
        int nsub = sub.size();
        str.write((char*)&nsub,sizeof(nsub));
        str.write((char*)&sub[0],sizeof(int)*nsub);
    }
}

void FractalMarkerSet::_fromStream(FractalMarkerSet &configuration, std::istream &str)
{
    str.read((char*)&configuration.mInfoType,sizeof(mInfoType));
    str.read((char*)&configuration._nmarkers,sizeof(_nmarkers));
    str.read((char*)&configuration._idExternal,sizeof(_idExternal));

    for(int i=0; i<configuration._nmarkers; i++)
    {
        //ID
        int id;
        str.read((char*)&id,sizeof(id));
        //NBITS
        int nbits;
        str.read((char*)&nbits,sizeof(nbits));

        //CORNERS
        std::vector<cv::Point3f> corners(4);
        str.read((char*)&corners[0],sizeof(cv::Point3f)*4);
        //MAT
        cv::Mat mat;
        mat.create(sqrt(nbits), sqrt(nbits), CV_8UC1);
        str.read((char*)mat.data, mat.elemSize() * mat.total());
        //SUBMARKERS
        int nsub;
        str.read((char*)&nsub,sizeof(nsub));
        std::vector<int> id_submarkers(nsub);
        if (nsub > 0)
            str.read((char*)&id_submarkers[0],sizeof(int)*nsub);

        FractalMarker fractalMarker = FractalMarker(id, mat, corners, id_submarkers);
        configuration.nbits_fractalMarkerIDs[mat.total()].push_back(fractalMarker.id);
        configuration.fractalMarkerCollection[fractalMarker.id] = fractalMarker;
    }

    //Add subfractals
    for(auto &id_marker:configuration.fractalMarkerCollection)
    {
        FractalMarker &marker = id_marker.second;
        for(auto id:id_marker.second.subMarkers())
            marker.addSubFractalMarker(configuration.fractalMarkerCollection[id]);
    }
}

void FractalMarkerSet::create(std::vector<std::pair<int,int>> regionsConfig, float pixSize)
{
    if(pixSize == -1)
    {
        mInfoType = NORM;
        pixSize = 1;
    }
    else
        mInfoType = PIX;

    _nmarkers = regionsConfig.size();
    _idExternal = 0;

    std::vector<int> submarkers;
    submarkers.clear();
    float pix = 0;
    for(int n=regionsConfig.size()-1; n>=0; n--)
    {
        int nVal = regionsConfig[n].first;
        int kVal = regionsConfig[n].second;

        cv::Mat mat = configureMat(nVal, kVal);

        pix = (nVal + 2) * pixSize;
        std::vector<cv::Point3f> corners = { cv::Point3f(-pix/2, pix/2, 0.),
                                             cv::Point3f(pix/2, pix/2, 0.),
                                             cv::Point3f(pix/2, -pix/2, 0.),
                                             cv::Point3f(-pix/2, -pix/2, 0.)
                                           };

        FractalMarker fractal = FractalMarker(n, mat, corners, submarkers);
        fractalMarkerCollection[n] = fractal;
        submarkers.clear();
        submarkers.push_back(n);


        float kValSup = regionsConfig[n-1].second - 2;
        float newP = (nVal+2)/(kValSup);
        pixSize = newP * pixSize;
    }

    //Normalize corners. Fractal marker: (-1,1,0)(1,1,0)(1,-1,0)(-1,-1,0)
    if(isNormalize()){
        for(auto &m:fractalMarkerCollection){
            for(auto &c:m.second.points)
            {
                c.x/=pix/2;
                c.y/=pix/2;
            }
        }
    }
}

cv::Mat FractalMarkerSet::configureMat(int nVal, int kVal, int maxIter)
{
    //Pixels to configure (n-k region)
    std::vector<cv::Point2i> pixels;
    for(int y=0; y<nVal; y++) {
        for(int x=0; x<nVal; x++) {
            if((x <= ((nVal-kVal)/2)-1 || x >= kVal+(nVal-kVal)/2)
                    || (y <= ((nVal-kVal)/2)-1 || y >= kVal+(nVal-kVal)/2)){
                pixels.push_back(cv::Point2i(x, y));
            }
        }
    }

    int dst_mkr = 0;
    int dst_set=0;

    cv::Mat m;
    int markerIters = 0;
    do{
        std::vector<cv::Point2i> conf_pixels = pixels;
        m = cv::Mat::ones(nVal, nVal, CV_8UC1);

//        struct timespec ts;
//        clock_gettime(CLOCK_MONOTONIC, &ts);
//        srand((time_t)ts.tv_nsec);

        //Random delete half of pixels
        int npix = conf_pixels.size()/2;
        for(int i=0; i<npix; i++) {
            int idxSelected = rand() % conf_pixels.size();
            conf_pixels.erase(conf_pixels.begin() + idxSelected);
        }

        //Put the new configuration (0 selected pixels)
        for(auto p:conf_pixels)
            m.at<char>(p.y,p.x) = 0;

        //Check marker distance to itself
        int new_dst_mkr = dstMarker(m);
        if(new_dst_mkr > dst_mkr)
        {
            //Check marker distance to set
            int new_dst_set = dstMarkerToFractalDict(m);
            if(new_dst_set > dst_set)
            {
                dst_mkr = new_dst_mkr;
                dst_set = new_dst_set;
            }
        }
    } while(markerIters++<maxIter);

    for(int y=((nVal-kVal)/2)+1; y<kVal+(nVal-kVal)/2-1; y++) {
        for(int x=((nVal-kVal)/2)+1; x<kVal+(nVal-kVal)/2-1; x++) {
                m.at<char>(y,x)=0;
            }
        }

    return m;
}

/*
 *  Calculate minimum distance between marker 'm' and the 4 rotations of each word in the dictionary
 */
int FractalMarkerSet::dstMarkerToFractalDict(cv::Mat m){
    int HDist = m.cols * m.rows; //distancia maxima
    int HDistTemp;

    for(auto marker:fractalMarkerCollection) {
        if(marker.second.nBits() == m.total())
        {
            HDistTemp = dstMarkerToMarker(marker.second.mat(), m);

            if(HDistTemp == 0)
                return 0;
            else if(HDistTemp < HDist)
                HDist = HDistTemp;
        }
    }
    return HDist;
}

/*
 *  Calculate minimum distance between marker1 'm' and the 4 rotations of marker2 'm2'
 */
auto rotate=[](const cv::Mat &m) {
    cv::Mat out;
    m.copyTo(out);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            out.at< char >(i, j) = m.at< char >(m.cols - j - 1, i);
        }
    }
    return out;
};

int FractalMarkerSet::dstMarker(const cv::Mat m) {

    int HDist = m.cols * m.rows;//maximum distance
    int HDistTemp;

    cv::Mat rot;
    m.copyTo(rot);


    for(int i=0; i<3; i++)
    {
        cv::Mat diff;
        rot=rotate(rot);
        cv::compare(rot, m, diff, cv::CMP_NE);
        HDistTemp = cv::countNonZero(diff);
        if(HDistTemp < HDist)
            HDist = HDistTemp;
    }
    return HDist;
}

int FractalMarkerSet::dstMarkerToMarker(const cv::Mat m1, const cv::Mat m2) {

    int HDist = m2.cols * m2.rows;//maximum distance
    int HDistTemp;

    for(int i=0; i<4; i++)
    {
        cv::Mat diff;
        cv::compare(rotate(m2), m1, diff, cv::CMP_NE);
        HDistTemp = cv::countNonZero(diff);
        if(HDistTemp < HDist)
            HDist = HDistTemp;
    }

    return HDist;
}


FractalMarkerSet FractalMarkerSet::readFromFile(std::string path)
{
    cv::FileStorage fs;
    try
    {
        fs.open(path, cv::FileStorage::READ);
    }
    catch (std::exception& ex)
    {
        throw cv::Exception(81818, "FractalMarkerSet::readFromFile", ex.what() + std::string(" file=)") + path, __FILE__, __LINE__);
    }

    FractalMarkerSet configuration;

    fs["mInfoType"] >> configuration.mInfoType;
    fs["fractal_levels"] >> configuration._nmarkers;
    fs["fractal_external_id"] >> configuration._idExternal;

    cv::FileNode markers = fs["markers"];
    int i = 0;
    for (cv::FileNodeIterator it = markers.begin(); it != markers.end(); ++it, i++)
    {
        int id = (*it)["id"];
        std::vector<int> bits;

        std::vector<cv::Point3f> corners;
        std::vector<int> submarkers_id;

        cv::FileNode bitsMarker = (*it)["bits"];
        for (cv::FileNodeIterator itb = bitsMarker.begin(); itb != bitsMarker.end(); ++itb)
        {
            bits.push_back(*itb);
        }
        cv::Mat m = cv::Mat(sqrt(bits.size()), sqrt(bits.size()), CV_32SC1);
        memcpy(m.data, bits.data(), bits.size()*sizeof(int));

        m.convertTo(m,CV_8UC1);

        cv::FileNode cornersFractal = (*it)["corners"];
        for (cv::FileNodeIterator itp = cornersFractal.begin(); itp != cornersFractal.end(); ++itp)
        {
            std::vector<float> coordinates3d;
            (*itp) >> coordinates3d;
            if (coordinates3d.size() != 3)
                throw cv::Exception(81818, "FractalMarkerSet::readFromFile", "invalid file type 3", __FILE__, __LINE__);
            cv::Point3f point(coordinates3d[0], coordinates3d[1], coordinates3d[2]);
            corners.push_back(point);
        }

        cv::FileNode submarkersID = (*it)["submarkers_id"];
        for (cv::FileNodeIterator itm = submarkersID.begin(); itm != submarkersID.end(); ++itm)
        {
            submarkers_id.push_back((*itm));
        }

        FractalMarker fractalMarker = FractalMarker(id, m, corners, submarkers_id);
        configuration.nbits_fractalMarkerIDs[m.total()].push_back(fractalMarker.id);
        configuration.fractalMarkerCollection[fractalMarker.id] = fractalMarker;
    }

    //Add subfractals
    for(auto &id_marker:configuration.fractalMarkerCollection)
    {
        FractalMarker &marker = id_marker.second;
        for(auto id:id_marker.second.subMarkers())
            marker.addSubFractalMarker(configuration.fractalMarkerCollection[id]);
    }

    return configuration;
}

    void FractalMarkerSet::saveToFile(cv::FileStorage& fs)
    {
        fs << "codeid" << "fractalmarkers";
        fs << "mInfoType" << mInfoType;
        fs << "fractal_levels" << _nmarkers;
        fs << "fractal_external_id" << _idExternal;
        fs << "markers"
           << "[";
        for(auto id_marker:fractalMarkerCollection) {

            FractalMarker marker = id_marker.second;

            fs << "{:"
               << "id" << (int)id_marker.first;

            fs << "bits"
                << "[:";
                cv::Mat m = marker.mat();
                for (int y = 0; y < m.cols; y++)
                    for (int x = 0; x < m.rows; x++)
                        if (m.at<char>(y, x) == 2)
                            fs << 0;
                        else
                            fs << m.at<char>(y, x);
            fs << "]";

            fs << "corners"
               << "[:";
                for (auto corner:marker.points)
                    fs << corner;
            fs << "]";

            fs << "submarkers_id"
               << "[:";
                for (auto idsub:marker.subMarkers())
                    fs << (int) idsub;
            fs << "]";

            fs << "}";
        }
        fs << "]";
    }

    bool FractalMarkerSet::isFractalMarker(cv::Mat &m, int nbits, int& markerid)
    {

        bool found = false;

        for(auto id:nbits_fractalMarkerIDs[nbits]){
            FractalMarker fm = fractalMarkerCollection[id];

            //Apply mask to substract submarkers
            cv::Mat m2;
            m.copyTo(m2, fm.mask());

            //Code without submarkers == fractal marker?
            if (cv::countNonZero(m2 != fm.mat()) == 0){
                found = true;

                //Change new code!!
                //////////////
                markerid = fm.id;
                /////////////

                break;
            }
        }

        return found;
    }

    FractalMarkerSet FractalMarkerSet::convertToMeters(float fractalSizeM)
    {
        if (!(isExpressedInPixels() || isNormalize()))
            throw cv::Exception(-1, "The FractalMarkers are not expressed in pixels", "FractalMarkerSet::convertToMeters", __FILE__,
                                __LINE__);

        FractalMarkerSet BInfo(*this);
        BInfo.mInfoType = FractalMarkerSet::METERS;

        // now, get the size of a pixel, and change scale
        float pixSizeM = fractalSizeM / float(BInfo.getFractalSize());

        for (size_t i=0; i < BInfo.fractalMarkerCollection.size(); i++)
        {
            //Convert to meters the position fractal marker
            for (int c = 0; c < 4; c++)
            {
                BInfo.fractalMarkerCollection[i][c] *= pixSizeM;
            }
        }
        return BInfo;
    }

    FractalMarkerSet FractalMarkerSet::normalize()
    {
        if (!(isExpressedInPixels() || isExpressedInMeters()))
            throw cv::Exception(-1, "The FractalMarkers are not expressed in pixels or meters", "FractalMarkerSet::convertToMeters", __FILE__,
                                __LINE__);

        FractalMarkerSet BInfo(*this);
        BInfo.mInfoType = FractalMarkerSet::NORM;

        float currentHalfSize = BInfo.getFractalSize()/2.f;

        for (size_t i=0; i < BInfo.fractalMarkerCollection.size(); i++)
        {
            //Normalize the position fractal marker
            for (size_t c = 0; c < 4; c++)
            {
                BInfo.fractalMarkerCollection[i][c] /= currentHalfSize;
            }
        }

        return BInfo;
    }

    std::map<int, std::vector<cv::Point3f>> FractalMarkerSet::getInnerCorners()
    {
        std::map<int, std::vector<cv::Point3f>> id_innerCorners;

        for(auto id_fm:fractalMarkerCollection)
        {
            int id = id_fm.first;
            FractalMarker fm = id_fm.second;

            for(auto ic:fm.getInnerCorners())
                id_innerCorners[id].push_back(ic); //Conversion

        }
        return id_innerCorners;
    }

    cv::Mat FractalMarkerSet::getFractalMarkerImage(int pixSize, bool border)
    {
        if(fractalMarkerCollection.size()<1)
            throw cv::Exception(9001, "There is not any fractal marker loaded",
                                "FractalMarkerSet::getFractalMarkerImage", __FILE__, __LINE__);

        //Smallest fractal marker
        FractalMarker innerM = (--fractalMarkerCollection.end())->second;
        float bitSize = innerM.getMarkerSize() / (pixSize * (sqrt(innerM.nBits())+2));

        FractalMarker externMarker = fractalMarkerCollection[_idExternal];
        float markerSize = externMarker.getMarkerSize()/bitSize;
        float markerBitSize = markerSize/(sqrt(externMarker.nBits())+2);
        cv::Mat img=cv::Mat::zeros(markerSize, markerSize, CV_8U);

        //Asign value pixels
        cv::Mat m = externMarker.mat();

        for (int y = m.cols-1; y >=0 ; y--)
            for (int x = m.cols-1; x >=0; x--){
                if (m.at<uchar>(y,x) == 1){
                    cv::Range val1 = cv::Range((1+y)*markerBitSize,(y+2)*markerBitSize);
                    cv::Range val2 = cv::Range((1+x)*markerBitSize,(x+2)*markerBitSize);

                    cv::Mat bit_pix=img(val1, val2);
                    bit_pix.setTo(cv::Scalar::all(255));
                }
            }

        for(auto idSubmarker:externMarker.subMarkers()){
            std::vector<int> idds;
            idds.push_back(idSubmarker);
            while(!idds.empty())
            {
                idSubmarker = idds.back();
                idds.pop_back();

                FractalMarker submarker = fractalMarkerCollection[idSubmarker];

                cv::Mat m = submarker.mat();
                cv::Point3f coord = submarker.points[0];

                float markerSize = submarker.getMarkerSize()/bitSize;
                float markerBitSize = markerSize/(sqrt(submarker.nBits())+2);

                //Get position inside fractal marker
                float offsetX = fabs(coord.x - externMarker.points[0].x)/bitSize ;
                float offsetY = fabs(coord.y - externMarker.points[0].y)/bitSize;

                //Asign value pixels
                for (int y = m.cols-1; y >=0 ; y--)
                    for (int x = m.cols-1; x >=0; x--){
                        if (m.at<uchar>(y,x) == 1){
                            cv::Range val1 = cv::Range((1+y)*markerBitSize + offsetY,(2+y)*markerBitSize + offsetY);
                            cv::Range val2 = cv::Range((1+x)*markerBitSize + offsetX,(2+x)*markerBitSize + offsetX);

                            cv::Mat bit_pix=img(val1, val2);
                            bit_pix.setTo(cv::Scalar::all(255));
                        }
                    }

                //Add submarkers to draw
                for(auto idsm:submarker.subMarkers())
                    idds.push_back(idsm);
            }
        }

        if(border)
            copyMakeBorder(img,img,markerBitSize,markerBitSize,markerBitSize,markerBitSize,cv::BORDER_CONSTANT,cv::Scalar::all(255));

        return img;
    }
};

```

6. src/fractallabelers/fractalmarkerset.h

```cpp
#include "fractalmarker.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <map>
#include "aruco_export.h"

namespace aruco {
    class ARUCO_EXPORT FractalMarkerSet{
    public:
        enum CONF_TYPES:
                uint64_t{
            FRACTAL_2L_6 = 0,
            FRACTAL_3L_6 = 1,
            FRACTAL_4L_6 = 2,
            FRACTAL_5L_6 = 3,
            CUSTOM=4 // for used defined dictionaries  (using load).
        };

        /**     create set of markers
         * @brief create
         * @param regionsConfig {N(f1),K(f1)}{N(f2):K(f2)}...{N(fn):K(fn)}
         * @param pixSize
         */
        void create(std::vector<std::pair<int,int>> regionsConfig, float pixSize);

        /**     configure bits of inner marker
         * @brief configureMat
         * @param nVal N region
         * @param kVal K region
         * @param maxIter Number of iteration
         * @return Mat configurated marker
         */
        cv::Mat configureMat(int nVal, int kVal, int maxIter=10000);
        // computes the distance of a marker to itself
        int dstMarker(const cv::Mat m);

        // computes distance between marker to marker
        int dstMarkerToMarker(const cv::Mat m1, const cv::Mat m2);

        // computes distance between marker to set of markers
        int dstMarkerToFractalDict(cv::Mat m);

        // saves to a binary stream
        static void _toStream(FractalMarkerSet &configuration, std::ostream &str);

        // load from a binary stream
        static void _fromStream(FractalMarkerSet &configuration, std::istream &str);

        static bool isPredefinedConfigurationString(std::string str);

        static std::string getTypeString(FractalMarkerSet::CONF_TYPES t);

        static CONF_TYPES getTypeFromString(std::string str);

        static FractalMarkerSet load(std::string info);

        static FractalMarkerSet loadPredefined(std::string info);

        static FractalMarkerSet loadPredefined(CONF_TYPES info);

        static FractalMarkerSet readFromFile(std::string path);

        // saves configuration to a text file
        void saveToFile(cv::FileStorage& fs);

        //Fractal configuration. id_marker
        std::map<int, FractalMarker> fractalMarkerCollection;
        //Nbits_idmarkers
        std::map<int, std::vector<int>> nbits_fractalMarkerIDs ;

        enum Fractal3DInfoType
        {
            NONE = -1,
            PIX = 0,
            METERS = 1,
            NORM = 2
        };  // indicates if the data in Fractal is expressed in meters or in pixels

        /**Indicates if the corners are expressed in meters
         */
        bool isExpressedInMeters() const
        {
            return mInfoType == METERS;
        }
        /**Indicates if the corners are expressed in meters
         */
        bool isExpressedInPixels() const
        {
            return mInfoType == PIX;
        }
        /**Indicates if the corners are normalized. -1..1 external marker
         */
        bool isNormalize() const
        {
            return mInfoType == NORM;
        }

        //Normalize fractal marker. The corners will go on to take the values (-1,1)(1,1),(1,-1)(-1,-1)
        FractalMarkerSet normalize();

        //Convert marker to meters
        FractalMarkerSet convertToMeters(float fractalSize_meters);

        static std::vector<std::string> getConfigurations();

        //Get fractal size (external marker)
        float getFractalSize() const
        {
            FractalMarker externalMarker = fractalMarkerCollection.at(_idExternal);
            return externalMarker.getMarkerSize();
        }

        //Get number of bits (external marker)
        int nBits() const
        {
            FractalMarker externalMarker = fractalMarkerCollection.at(_idExternal);
            return externalMarker.nBits();
        }

        // Check if m is a inner marker, and get its id.
        bool isFractalMarker(cv::Mat &m, int nbits, int&id);

        // Get all inners corners
        std::map<int, std::vector<cv::Point3f>> getInnerCorners();

        cv::Mat getFractalMarkerImage(int pixSize, bool border=false);

        // variable indicates if the data is expressed in meters or in pixels or are normalized
        int mInfoType;/* -1:NONE, 0:PIX, 1:METERS, 2:NORMALIZE*/

    private:
        // Number of levels
        int _nmarkers;
        //ID external marker
        int _idExternal=0;
        //Configuration dictionary
        std::string config;
    };
}
```

7. src/fractallabelers/fractalposetracker.cpp

```cpp

#include "fractalposetracker.h"
#include "../levmarq.h"
#include "../ippe.h"
#include "../aruco_export.h"
#include "../timers.h"


#include "opencv2/calib3d/calib3d.hpp"
#include "../aruco_cvversioning.h"

namespace aruco {

    /*
     * KeyPoint cornersubpixel
     */
    void kcornerSubPix(const cv::Mat image, std::vector<cv::KeyPoint> &kpoints)
    {
        std::vector<cv::Point2f> points;
        cv::KeyPoint::convert(kpoints, points);

        cv::Size winSize = cv::Size(4, 4);
        cv::Size zeroZone = cv::Size( -1, -1 );
        cv::TermCriteria criteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);      //cv::cornerSubPix(image, points, winSize, zeroZone, criteria);
        cornerSubPix(image, points, winSize, zeroZone, criteria);

        //Update kpoints
        uint32_t i=0;
        for(auto &k:kpoints){
            k.pt = points[i];
            i++;
        }
    }

    /*
     * KeyPoints Filter. Delete kpoints with low response and duplicated.
     */
    void kfilter(std::vector<cv::KeyPoint> &kpoints)
    {
        float minResp = kpoints[0].response;
        float maxResp = kpoints[0].response;
        for (auto &p:kpoints){
            p.size=40;
            if(p.response < minResp) minResp = p.response;
            if(p.response > maxResp) maxResp = p.response;
        }
        float thresoldResp = (maxResp - minResp) * 0.20f + minResp;

        //Erase kepoints with low response (20%)
        for (uint32_t i=0;i< kpoints.size(); i++)
            if(kpoints[i].response < thresoldResp){
                kpoints.erase(kpoints.begin()+i);
                i--;
            }

        //Duplicated keypoints (closer)
        for(uint32_t xi=0; xi<kpoints.size();xi++)
            for(uint32_t xj=xi+1; xj<kpoints.size();xj++)
            {
                if(pow(kpoints[xi].pt.x - kpoints[xj].pt.x,2) + pow(kpoints[xi].pt.y - kpoints[xj].pt.y,2) < 200)
                    kpoints.erase(kpoints.begin()+xj--);
            }
    }

    FractalPoseTracker::FractalPoseTracker()
    {

    }

    void FractalPoseTracker::setParams(const CameraParameters& cam_params, const FractalMarkerSet& msconf, const float markerSize)
    {
        _cam_params = cam_params;
        _fractalMarker = msconf;

        if (!cam_params.isValid())
            throw cv::Exception(9001, "Invalid camera parameters", "FractalPoseTracker::setParams", __FILE__,
                                __LINE__);
        if (_fractalMarker.mInfoType == aruco::FractalMarkerSet::NONE)
            throw cv::Exception(9001, "Invalid FractalMarker", "FractalPoseTracker::setParams", __FILE__, __LINE__);
        if ((_fractalMarker.mInfoType == aruco::FractalMarkerSet::PIX ||
               _fractalMarker.mInfoType == aruco::FractalMarkerSet::NORM) && markerSize <= 0)
            throw cv::Exception(9001, "You should indicate the markersize since the Fractal Marker is in pixels or normalized",
                                "FractalPoseTracker::setParams", __FILE__, __LINE__);
        if (_fractalMarker.mInfoType == aruco::FractalMarkerSet::PIX ||
               _fractalMarker.mInfoType == aruco::FractalMarkerSet::NORM)
            _fractalMarker = _fractalMarker.convertToMeters(markerSize);

        for(auto id_innerMarker:_fractalMarker.fractalMarkerCollection)
        {
            int markerId = id_innerMarker.first;
            FractalMarker innerMarker = id_innerMarker.second;

            //Inner corners
            _id_innerp3d[markerId] = innerMarker.getInnerCorners();
            for(auto pt : _id_innerp3d[markerId])
            {
                _innerp3d.push_back(pt);
                _innerkpoints.push_back(cv::KeyPoint(cv::Point2f(pt.x, pt.y), 20, -1, 0, markerId));
            }

            //radius search by marker
            float radiusF = 0.25;
            float ratio=innerMarker.getMarkerSize()/_fractalMarker.getFractalSize();
            float NBits=float(sqrt(_fractalMarker.nBits())+2)*ratio;
            _id_radius[markerId] = ((2*NBits)/((sqrt(innerMarker.nBits())+2)*(sqrt(_fractalMarker.nBits())+2)))*radiusF*_fractalMarker.getFractalSize()/2;
        }

        //Get synthetic image (pixsize 6 in order to get inner corners classification)
        cv::Mat imageGray = _fractalMarker.getFractalMarkerImage(6) * 255;
        assignClass(imageGray, _innerkpoints, true);
        _kdtree.build(_innerkpoints);

//        #define _fractal_debug_classification
        #ifdef _fractal_debug_classification
        drawKeyPoints(imageGray,_innerkpoints, false, true);
        #endif
    }

    bool FractalPoseTracker::fractalRefinement(const cv::Ptr<MarkerDetector> markerDetector, int markerWarpPix)
    {
        //ScopedTimerEvents Timer("fractal-refinement");

        std::vector<cv::Mat> imagePyramid = markerDetector->getImagePyramid();
        std::vector<cv::Point3f> _ref_inner3d;
        std::vector<cv::Point2f> _ref_inner2d;

        cv::Mat _p_rvec;
        _rvec.copyTo(_p_rvec);
        cv::Mat _p_tvec;
        _tvec.copyTo(_p_tvec);

        cv::Mat rot;
        cv::Rodrigues(_rvec, rot);

        for (auto id_marker : _fractalMarker.fractalMarkerCollection)
        {
            //Check z value for 4 external corners
            std::vector<cv::Point3f> marker3d;
            for(auto pt : id_marker.second.points)
            {
                cv::Mat_<double> src(3,1,rot.type());
                src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;

                cv::Mat  cam_image_point = rot * src + _tvec;
                cam_image_point = cam_image_point/cv::norm(cam_image_point);

                if(cam_image_point.at<double>(2,0)>0.85)
                    marker3d.push_back(pt);
                else break;
            }

            std::vector<cv::Point3f> _inners3d;
            std::vector<cv::Point2f> _inners2d;
            float area=0;
            if(marker3d.size()<4)
            {
                if(_id_area.find(id_marker.first) != _id_area.end())
                    area = _id_area[id_marker.first];
				else
					return false;

                for(auto pt : _id_innerp3d[id_marker.first])
                {
                    cv::Mat_<double> src(3,1,_rvec.type());
                    src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;

                    cv::Mat  cam_image_point = rot * src +_tvec;
                    cam_image_point = cam_image_point/cv::norm(cam_image_point);

                    if(cam_image_point.at<double>(2,0)>0.85)
                        _inners3d.push_back(pt);
                }

                if(_inners3d.size()==0)
                   return false;

                cv::projectPoints(_inners3d, _rvec, _tvec,
                                  _cam_params.CameraMatrix, _cam_params.Distorsion, _inners2d);
            }
            else
            {
                std::vector<cv::Point2f> marker2d;
                cv::projectPoints(marker3d, _rvec, _tvec,
                                  _cam_params.CameraMatrix, _cam_params.Distorsion, marker2d);

                cv::Point2f v01 = marker2d[1] - marker2d[0];
                cv::Point2f v03 = marker2d[3] - marker2d[0];
                float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
                cv::Point2f v21 = marker2d[1] - marker2d[2];
                cv::Point2f v23 = marker2d[3] - marker2d[2];
                float area2 = fabs(v21.x * v23.y - v21.y * v23.x);

                area = (area2 + area1) / 2.f;
                _id_area[id_marker.first] = area;

                _inners3d = _id_innerp3d[id_marker.first];
                cv::projectPoints(_inners3d, _rvec, _tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, _inners2d);
            }

            size_t imgPyrIdx = 0;
            auto markerWarpSize = (sqrt(id_marker.second.nBits())+2)*markerWarpPix;
            float desiredarea = std::pow(static_cast<float>(markerWarpSize), 2.f);
            for (size_t p = 1; p < imagePyramid.size(); p++)
            {
                if (area/ pow(4, p) >= desiredarea) imgPyrIdx = p;
                else break;
            }

            float ratio = float(imagePyramid[imgPyrIdx].cols)/float(imagePyramid[0].cols);

            //std::cout << "REFINE["<< id_marker.first <<"], imgPyrId:"<<imgPyrIdx << ", ratio:"<< ratio<<std::endl;

            std::vector<double> _inners2d_error;
            if (ratio == 1 && area >= desiredarea){
                int halfwsize= 4*float(imagePyramid[imgPyrIdx].cols)/float(imagePyramid[imgPyrIdx].cols) +0.5 ;

                std::vector<cv::Point2f> _inners2d_copy;
                for(auto pt:_inners2d){ _inners2d_copy.push_back(pt);}
                cornerSubPix(imagePyramid[imgPyrIdx], _inners2d, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
                int idx=0;
                for(auto pt:_inners2d)
                {
                    _inners2d_error.push_back(sqrt(pow(_inners2d_copy[idx].x - pt.x,2) + pow(_inners2d_copy[idx].y - pt.y,2)));
                    idx++;
                }
            }
            else if (ratio != 1 && area >= desiredarea){

                std::vector<cv::Point2f> _inners2d_copy;
                for(auto &pt:_inners2d){ _inners2d_copy.push_back(pt); pt *=ratio;}

                std::vector<std::vector<cv::Point2f>> vpnts;
                vpnts.push_back(_inners2d);
                markerDetector->cornerUpsample(vpnts, imagePyramid[imgPyrIdx].size());

                int idx=0;
                for(auto pt:vpnts[0])
                {
                    _inners2d_error.push_back(sqrt(pow(_inners2d_copy[idx].x - pt.x,2) + pow(_inners2d_copy[idx].y - pt.y,2)));
                    _inners2d[idx++] = pt;
                }
            }

            //Elimina puntos que no son esquinas
            if (area >= desiredarea) {
                //We discard outliers. Points above limit Q3+3*(Q3-1)
                std::vector<double>_inners2d_error_copy;
                for(auto err:_inners2d_error)
                    _inners2d_error_copy.push_back(err);
                sort(_inners2d_error_copy.begin(), _inners2d_error_copy.end());
                int q1 = (_inners2d_error_copy.size()+1)/4;
                int q3 = 3*(_inners2d_error_copy.size()+1)/4;

                double limit = _inners2d_error_copy[q3] + 3*(_inners2d_error_copy[q3]-_inners2d_error_copy[q1]);

                int wsize = 10;
                for(int idx=0; idx<_inners2d.size(); idx++)
                {
                    if(_inners2d_error[idx]>limit)
                        continue;

                    float x= int(_inners2d[idx].x+0.5f);
                    float y= int(_inners2d[idx].y+0.5f);

                    cv::Rect r= cv::Rect(x-wsize,y-wsize,wsize*2+1,wsize*2+1);
                    //Check boundaries
                    if(r.x<0 || r.x+r.width>imagePyramid[0].cols || r.y<0 ||
                            r.y+r.height>imagePyramid[0].rows) continue;

                    int endX=r.x+r.width;
                    int endY=r.y+r.height;
                    uchar minV=255,maxV=0;
                    for(int y=r.y; y<endY; y++){
                        const uchar *ptr=imagePyramid[0].ptr<uchar>(y);
                        for(int x=r.x; x<endX; x++)
                        {
                            if(minV>ptr[x]) minV=ptr[x];
                            if(maxV<ptr[x]) maxV=ptr[x];
                        }
                    }

                    if ((maxV-minV) < 25) continue;

                    _ref_inner3d.push_back(_inners3d[idx]);
                    _ref_inner2d.push_back(_inners2d[idx]);
                }
            }
            //Timer.add("refine-"+std::to_string(id_marker.first));
        }

        //Solve
        if(_ref_inner3d.size()>4)
        {
            aruco::solvePnP(_ref_inner3d, _ref_inner2d, _cam_params.CameraMatrix, _cam_params.Distorsion,_rvec,_tvec);

//#define _fractal_debug_inners
#ifdef _fractal_debug_inners
            cv::Mat InImageCopy;
            cv::cvtColor(imagePyramid[0], InImageCopy, CV_GRAY2RGB);

            //Show first position (estimation with detected markers)
            std::vector<cv::Point2f> preinnersPrj;
            for (auto id_marker : _fractalMarker.fractalMarkerCollection)
            {
                cv::projectPoints(_id_innerp3d[id_marker.first], _p_rvec, _p_tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, preinnersPrj);
                for(auto pt:preinnersPrj)
                    cv::circle(InImageCopy, pt, 5, cv::Scalar(0,0,255),CV_FILLED);
            }

            //Show first position with refinement
            for(auto p:_ref_inner2d)
                cv::circle(InImageCopy, p, 5, cv::Scalar(255,0,0),CV_FILLED);

            _rvec.convertTo(_rvec,CV_32F);
            _tvec.convertTo(_tvec,CV_32F);
            cv::Rodrigues(_rvec, rot);

            std::vector<cv::Point3f> _inners;
            for(auto pt : _inner_corners_3d)
            {
                cv::Mat_<float> src(3,1,rot.type());
                src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;

                cv::Mat  cam_image_point = rot * src + _tvec ;
                cam_image_point = cam_image_point/cv::norm(cam_image_point);

                if(cam_image_point.at<float>(2,0)>0.85f)
                    _inners.push_back(pt);
            }
            std::vector<cv::Point2f> _inners_prj;
            cv::projectPoints(_inners, _rvec, _tvec,
                              _cam_params.CameraMatrix, _cam_params.Distorsion, _inners_prj);

            //Show new projection using all inner points
            for(auto pt:_inners_prj)
                cv::circle(InImageCopy, pt, 5, cv::Scalar(0,255,0),CV_FILLED);

            cv::namedWindow("AA",cv::WINDOW_NORMAL);
            imshow("AA", InImageCopy);
            cv::waitKey();
#endif
            //Timer.add("solve");

            return true;
        }
        else
            return false;
    }

    bool FractalPoseTracker::fractalInnerPose(const cv::Ptr<aruco::MarkerDetector> markerDetector, const std::vector<aruco::Marker>& vmarkers, bool refinement)
    {
        if(vmarkers.size()>0)
        {
//            ScopedTimerEvents Timer("pnp");
//            std::cout << "[Case 1]"<< std::endl;
            std::vector<cv::Point2f> p2d;
            std::vector<cv::Point3f> p3d;
            for (auto marker : vmarkers)
            {
                if (_fractalMarker.fractalMarkerCollection.find(marker.id) != _fractalMarker.fractalMarkerCollection.end())
                {
                    for (auto p : marker)
                        p2d.push_back(p);

                    for (auto p : _fractalMarker.fractalMarkerCollection[marker.id].points)
                        p3d.push_back(p);
                }
            }

            //Initial pose estimation
            aruco::solvePnP(p3d, p2d, _cam_params.CameraMatrix, _cam_params.Distorsion,_rvec,_tvec);

//            Timer.add("solve");

            //REFINE
            if(refinement) {
                fractalRefinement(markerDetector);
//                Timer.add("refine-solution");
            }

            return true;
        }
        else
        {
            if(!_rvec.empty())
            {
//                std::cout << "[Case 2]"<< std::endl;
//                ScopedTimerEvents Timer("ransac");
//                Timer.add("detect");

                std::vector<cv::Point2f> innerPoints2d;
                std::vector<cv::Point3f> innerPoints3d;

                float radius=0;
                cv::Mat rot;
                cv::Rodrigues(_rvec, rot);

                //Getting the keypoints search radius
                for(auto id_marker:_fractalMarker.fractalMarkerCollection)
                {
                    std::vector<cv::Point2f> marker2d;
                    std::vector<cv::Point3f> marker3d;
                    for(auto pt:_fractalMarker.fractalMarkerCollection[id_marker.first].points)
                    {
                        cv::Mat_<double> src(3,1,rot.type());
                        src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;
                        cv::Mat  cam_image_point = rot * src + _tvec ;
                        cam_image_point = cam_image_point/cv::norm(cam_image_point);

                        if(cam_image_point.at<double>(2,0)>0.85)
                            marker3d.push_back(pt);
                        else break;
                    }

                    if(marker3d.size()==4)
                    {
                        cv::projectPoints(marker3d, _rvec, _tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, marker2d);

                        //Find marker area
                        cv::Point2f v01 = marker2d[1] - marker2d[0];
                        cv::Point2f v03 = marker2d[3] - marker2d[0];
                        float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
                        cv::Point2f v21 = marker2d[1] - marker2d[2];
                        cv::Point2f v23 = marker2d[3] - marker2d[2];
                        float area2 = fabs(v21.x * v23.y - v21.y * v23.x);
                        double area = (area2 + area1) / 2.f;

                        auto markerWarpSize = (sqrt(_fractalMarker.fractalMarkerCollection[id_marker.first].nBits())+2)*10;
                        float desiredarea = std::pow(static_cast<float>(markerWarpSize), 2.f);
                        if(area >= desiredarea)
                        {
                            for(auto pt:_id_innerp3d[id_marker.first])
                                innerPoints3d.push_back(pt);
                        }

                        if(radius == 0.f)
                            radius = sqrt(area)/(sqrt(id_marker.second.nBits())+2.f);
                    }
                    else
                    {
                        for(auto pt:_id_innerp3d[id_marker.first])
                            innerPoints3d.push_back(pt);
                    }
                }

                if(radius==0) radius = _preRadius;
                _preRadius = radius;

                if(innerPoints3d.size() > 0 && radius > 0)
                {
                    cv::projectPoints(innerPoints3d, _rvec, _tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, innerPoints2d);

                    cv::Mat region;
                    cv::Point2f offset;
                    float ratio;
                    if(!ROI(markerDetector->getImagePyramid(), region, innerPoints2d, offset, ratio))
                        return false;
                    radius = radius * ratio;
//                    Timer.add("roi");


                    std::cout << "radius: " << radius << std::endl;

//#define _fractal_debug_region
#ifdef _fractal_debug_region
                    cv::Mat out;
                    cv::cvtColor(region, out, CV_GRAY2RGB);
                    for(uint32_t i=0; i<innerPoints2d.size(); i++)
                        circle(out, innerPoints2d[i], radius,  cv::Scalar(0,0,255),2);
                    cv::imshow("REGION ", out);
                    cv::waitKey();
#endif

                    //FAST
                    std::vector<cv::KeyPoint> kpoints;
                    cv::Ptr<cv::FastFeatureDetector> fd = cv::FastFeatureDetector::create();
                    fd->detect(region, kpoints);
//                    Timer.add("fast");


                    if(kpoints.size() > 0)
                    {
                        //Filter kpoints (low response) and removing duplicated.
                        kfilter(kpoints);
//                        Timer.add("fast-filter");
//                        std::cout << "fast-filter" << std::endl;

                        //Assign class to keypoints
                        assignClass(region, kpoints);
//                        Timer.add("fast-class");

//#define _fractal_debug_classification
#ifdef _fractal_debug_classification
                        drawKeyPoints(region, kpoints);
                        cv::waitKey();
#endif
                        //Get keypoints with better response in a radius
                        picoflann::KdTreeIndex<2,PicoFlann_KeyPointAdapter>  kdtreeImg;
                        kdtreeImg.build(kpoints);
                        std::vector<std::pair<uint, std::vector<uint>>> inner_candidates;
                        for(uint idx=0; idx<innerPoints2d.size(); idx++){
                            std::vector<std::pair<uint32_t, double>> res = kdtreeImg.radiusSearch(kpoints, innerPoints2d[idx], radius);

                            std::vector<uint> candidates;
                            for(auto r:res){
                                if(kpoints[r.first].class_id == _innerkpoints[idx].class_id)
                                    candidates.push_back(r.first);
                            }
                            if(candidates.size() > 0)
                                inner_candidates.push_back(make_pair(idx, candidates));
                        }

//                        Timer.add("candidates");
                        cv::Mat bestModel = fractal_solve_ransac(innerPoints2d.size(), inner_candidates, kpoints);
//                        Timer.add("find-solution");

                        if(!bestModel.empty()){
                            std::vector<cv::Point3f>p3d;
                            std::vector<cv::Point2f>p2d;

                            std::vector<cv::Point2f> pnts, pntsDst;
                            cv::KeyPoint::convert(kpoints, pnts);
                            perspectiveTransform(pnts, pntsDst, bestModel);
                            for(uint32_t id=0; id<pntsDst.size(); id++)
                            {
                                std::vector<std::pair<uint32_t, double>> res = _kdtree.radiusSearch(_innerkpoints, pntsDst[id], _id_radius[0]);

                                int i=0;
                                for(auto r:res)
                                {
                                    uint32_t innerId = r.first;
                                    double dist = sqrt(r.second);

                                    uint32_t fmarkerId = _innerkpoints[innerId].octave;
                                    if(dist > _id_radius[fmarkerId])
                                        res.erase(res.begin()+i);
                                    else i++;
                                }
                                if(res.size()>0){
                                    p3d.push_back(_innerp3d[res[0].first]);
                                    p2d.push_back(cv::Point2f(kpoints[id].pt.x + offset.x, kpoints[id].pt.y + offset.y)/ratio);
                                }
                            }

                            if(p3d.size() >= 4)
                            {
//                                std::cout << "solves" << std::endl;
                                aruco::solvePnP(p3d, p2d, _cam_params.CameraMatrix, _cam_params.Distorsion, _rvec, _tvec);
//                                Timer.add("solves");

                                if(refinement) {
//                                    std::cout << "refine-solution" << std::endl;
                                    fractalRefinement(markerDetector);
//                                    Timer.add("refine-solution");
                                }

                                return true;
                            }
                        }
                    }
                }
            }
            _rvec = cv::Mat();
            _tvec = cv::Mat();

            return false;
       }
    }

    bool FractalPoseTracker::ROI(const std::vector<cv::Mat> imagePyramid, cv::Mat &img, std::vector<cv::Point2f> &innerPoints2d, cv::Point2f &offset, float &ratio)
    {
        cv::Mat rot;
        cv::Rodrigues(_rvec, rot);

        //Biggest marker projection
        std::vector<cv::Point2f> biggest_p2d;
        std::vector<cv::Point3f> biggest_p3d;
        for(int idx=0; idx<_fractalMarker.fractalMarkerCollection.size()&biggest_p3d.size()<4; idx++)
        {
            biggest_p3d.empty();
            for(auto pt:_fractalMarker.fractalMarkerCollection[idx].points)
            {
                cv::Mat_<double> src(3,1,rot.type());
                src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;
                cv::Mat  cam_image_point = rot * src + _tvec ;
                cam_image_point = cam_image_point/cv::norm(cam_image_point);

                if(cam_image_point.at<double>(2,0)>0.85)
                    biggest_p3d.push_back(pt);
                else break;
            }
        }

        if(!biggest_p3d.empty())
        {
            cv::projectPoints(biggest_p3d, _rvec, _tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, biggest_p2d);

            //Smallest marker projection
            std::vector<cv::Point2f> smallest_p2d;
            std::vector<cv::Point3f> smallest_p3d;
            auto marker_smallest = (--_fractalMarker.fractalMarkerCollection.end())->second;
            for(auto pt:marker_smallest.points)
                smallest_p3d.push_back(pt);
            cv::projectPoints(smallest_p3d, _rvec, _tvec, _cam_params.CameraMatrix, _cam_params.Distorsion, smallest_p2d);

            //Smallest marker area
            cv::Point2f v01 = smallest_p2d[1] - smallest_p2d[0];
            cv::Point2f v03 = smallest_p2d[3] - smallest_p2d[0];
            float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
            cv::Point2f v21 = smallest_p2d[1] - smallest_p2d[2];
            cv::Point2f v23 = smallest_p2d[3] - smallest_p2d[2];
            float area2 = fabs(v21.x * v23.y - v21.y * v23.x);
            double area = (area2 + area1) / 2.f;


            //Compute boundaries region
            float border = sqrt(area)/sqrt(marker_smallest.nBits())+2;
            int minX=imagePyramid[0].cols, minY=imagePyramid[0].rows, maxX=0, maxY=0;
            for(auto p:biggest_p2d)
            {
                if(p.x < minX) minX = p.x-border;
                if(p.x > maxX) maxX = p.x+border;
                if(p.y < minY) minY = p.y-border;
                if(p.y > maxY) maxY = p.y+border;
            }
            if(minX < 0) minX=0;
            if(minY < 0) minY=0;
            if(maxX > imagePyramid[0].cols) maxX=imagePyramid[0].cols;
            if(maxY > imagePyramid[0].rows) maxY=imagePyramid[0].rows;


            //Select imagePyramid
            size_t imgPyrIdx = 0;
            auto markerWarpSize = (sqrt(marker_smallest.nBits())+2)*10;
            float desiredarea = std::pow(static_cast<float>(markerWarpSize), 2.f);
            for (size_t p = 1; p < imagePyramid.size(); p++)
            {
                if (area / pow(4, p) >= desiredarea) imgPyrIdx = p;
                else break;
            }

            ratio=float(imagePyramid[imgPyrIdx].cols)/float(imagePyramid[0].cols);
            offset=cv::Point2i(minX, minY)*ratio;
            cv::Rect region = cv::Rect(cv::Point2i(minX, minY)*ratio , cv::Point2i(maxX, maxY)*ratio);
            img = imagePyramid[imgPyrIdx](region);

            for(auto &pt:innerPoints2d){
                pt.x = pt.x*ratio - region.x;
                pt.y = pt.y*ratio - region.y;
            }
            return true;
        }
        else
            return false;
    }

    cv::Mat FractalPoseTracker::fractal_solve_ransac(int ninners, std::vector<std::pair<uint, std::vector<uint>>> inner_kpnt, std::vector<cv::KeyPoint> kpnts, uint32_t maxIter, float _minInliers, float _thresInliers)
    {
        std::vector<cv::Point2f> pnts;
        cv::KeyPoint::convert(kpnts, pnts);

        //Number randomly values selected
        uint32_t numInliers=4;
        //Number of inliers to consider it good model, stop iterating!!!
        uint32_t thresInliers = uint32_t(ninners*_thresInliers);
        //Enough number of inliers to consider the model as valid
        uint32_t minInliers = uint32_t(ninners*_minInliers);

        //Number of inliers
        uint32_t bestInliers = 0;
        //Best model
        cv::Mat bestH = cv::Mat();

//        struct timespec ts;
//        clock_gettime(CLOCK_MONOTONIC, &ts);
//        srand((time_t)ts.tv_nsec);

        uint32_t count=0;
        for(auto ik:inner_kpnt)
            if(ik.second.size()>0)
                count++;
        if(count < minInliers) return cv::Mat();

        uint32_t iter = 0;
        do
        {
            // New stop condition to avoid infinite loop, when it tries to find the initial inliers
            // For instance, for innerkpts group: 23{2} 18{3}, 14{3}, 3{7,6,5}, 2{3},
            // when the initial group of inliers is selected:
            // 23{2},18{3},3{6} there's no way to find the fourth ...
            uint32_t iter2 = 0;
            std::vector<std::pair<uint, uint>> inliers;
            while(inliers.size() < numInliers && iter2++<maxIter)
            {
                uint idx = rand()%inner_kpnt.size();
                uint id_dst = inner_kpnt[idx].first;

                uint idxc = rand()%inner_kpnt[idx].second.size();
                uint id_src = inner_kpnt[idx].second[idxc];

                //avoid duplicate observations
                bool exist=false;
                for(auto in:inliers)
                {
                    if( (in.first != id_dst) && (in.second !=id_src))
                        continue;
                    else {
                        exist=true;
                        break;
                    }
                }

                //if it is a new observation, add it!
                if(!exist) inliers.push_back(std::make_pair(id_dst, id_src));
            }

            if(iter2 >= maxIter)
                return cv::Mat();

            std::vector<cv::Point2f> srcInliers;
            std::vector<cv::Point2f> dstInliers;
            for(auto in:inliers){
                dstInliers.push_back(_innerkpoints[in.first].pt);
                srcInliers.push_back(kpnts[in.second].pt);
            }
            //Fit model with initial random inliers
            cv::Mat H = findHomography(srcInliers,  dstInliers);

            if(!H.empty())
            {
                std::vector<std::pair<uint, uint>> newInliers;
                std::vector<std::pair<uint, uint>> newInliers2;

                std::vector<cv::Point2f> dstNewInliers;
                std::vector<cv::Point2f> srcNewInliers;
                std::vector<cv::Point2f> pntsTranf;
                perspectiveTransform(pnts, pntsTranf, H);

                for(uint idP=0; idP< pntsTranf.size(); idP++)
                {
                    std::vector<std::pair<uint32_t,double>>res = _kdtree.radiusSearch(_innerkpoints, pntsTranf[idP], _id_radius[0]);

                    int i=0;
                    for(auto r:res)
                    {
                        uint32_t innerId = r.first;
                        double dist = sqrt(r.second);

                        uint32_t fmarkerId = _innerkpoints[innerId].octave;
                        if(dist > _id_radius[fmarkerId])
                            res.erase(res.begin()+i);
                        else i++;
                    }

                    if(res.size() > 0){
                        //avoid duplicate observations
                        bool exist=false;
                        for(auto in:newInliers)
                        {
                            if(in.first != res[0].first)
                                continue;
                            else {
                                exist=true;
                                break;
                            }
                        }
                        //if it is a new observation and these belong to the same class, add it!
                        if(!exist)
                        {
                            if(_innerkpoints[res[0].first].class_id == kpnts[idP].class_id)
                                newInliers.push_back(std::make_pair(res[0].first, idP));
                        }
                    }
                }
                if(newInliers.size() > bestInliers)
                {
                    bestInliers = newInliers.size();
                    bestH = H;
                }
            }
            iter++;
        } while(iter<maxIter && bestInliers<thresInliers);

//        std::cout << "[RANSAC] minInliers: "<< minInliers ;
//        std::cout << " ,bestInliers: "<< bestInliers;
//        std::cout << " ,iterations: "<< iter << std::endl;

        if(bestInliers<minInliers)
            bestH = cv::Mat();

        return bestH;
    }

    void FractalPoseTracker::drawKeyPoints(const cv::Mat image, std::vector<cv::KeyPoint> kpoints, bool text, bool transf)
    {
        if(transf){
            //Convert point range from norm (-size/2, size/2) to (0,imageSize)
            for(auto &k:kpoints){
                k.pt.x = image.cols * (k.pt.x/_fractalMarker.getFractalSize() + 0.5f);
                k.pt.y = image.rows * (-k.pt.y/_fractalMarker.getFractalSize() + 0.5f);
            }
        }

        cv::Mat out;
        cv::cvtColor(image, out, CV_GRAY2BGR);
        //drawKeypoints(image, kpoints, out, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::Scalar color;
        for(auto kp:kpoints)
        {
            if(kp.class_id == -1 ) color = cv::Scalar(255,0,255);
            if(kp.class_id == 0 ) color = cv::Scalar(0,0,255);
            if(kp.class_id == 1 ) color = cv::Scalar(0,255,0);
            if(kp.class_id == 2 ) color = cv::Scalar(255,0,0);
            circle(out, kp.pt, 2, color,-1);
        }

        if(text){
            int nkm=0;
            for(auto kp:kpoints)
            {
                putText( out, std::to_string(nkm++), kp.pt, CV_FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8 );
            }
        }

//#ifdef _DEBUG
//        imshow("KPoints", out);
//        cv::waitKey();
//#endif
    }

    void FractalPoseTracker::assignClass(const cv::Mat &im, std::vector<cv::KeyPoint>& kpoints, bool transf, int wsize)
    {
        if(im.type()!=CV_8UC1)
            throw std::runtime_error("assignClass Input image must be 8UC1");
        int wsizeFull=wsize*2+1;

        cv::Mat labels = cv::Mat::zeros(wsize*2+1,wsize*2+1,CV_8UC1);
        cv::Mat thresIm=cv::Mat(wsize*2+1,wsize*2+1,CV_8UC1);

        for(auto &kp:kpoints)
        {
            float x = kp.pt.x;
            float y = kp.pt.y;

            //Convert point range from norm (-size/2, size/2) to (0,imageSize)
            if(transf){
                x = im.cols * (x/_fractalMarker.getFractalSize() + 0.5f);
                y = im.rows * (-y/_fractalMarker.getFractalSize() + 0.5f);
            }

            x= int(x+0.5f);
            y= int(y+0.5f);

            cv::Rect r= cv::Rect(x-wsize,y-wsize,wsize*2+1,wsize*2+1);
            //Check boundaries
            if(r.x<0 || r.x+r.width>im.cols || r.y<0 ||
                    r.y+r.height>im.rows) continue;

            int endX=r.x+r.width;
            int endY=r.y+r.height;
            uchar minV=255,maxV=0;
            for(int y=r.y; y<endY; y++){
                const uchar *ptr=im.ptr<uchar>(y);
                for(int x=r.x; x<endX; x++)
                {
                    if(minV>ptr[x]) minV=ptr[x];
                    if(maxV<ptr[x]) maxV=ptr[x];
                }
            }

            if ((maxV-minV) < 25) {
                kp.class_id=0;
                continue;
            }

            double thres=(maxV+minV)/2.0;

            unsigned int nZ=0;
            //count non zero considering the threshold
            for(int y=0; y<wsizeFull; y++){
                const uchar *ptr=im.ptr<uchar>( r.y+y)+r.x;
                uchar *thresPtr= thresIm.ptr<uchar>(y);
                for(int x=0; x<wsizeFull; x++){
                    if( ptr[x]>thres) {
                        nZ++;
                        thresPtr[x]=255;
                    }
                    else thresPtr[x]=0;
                }
            }
            //set all to zero labels.setTo(cv::Scalar::all(0));
            for(int y=0; y<thresIm.rows; y++){
                uchar *labelsPtr=labels.ptr<uchar>(y);
                for(int x=0; x<thresIm.cols; x++) labelsPtr[x]=0;
            }

            uchar newLab = 1;
            std::map<uchar, uchar> unions;
            for(int y=0; y<thresIm.rows; y++){
                uchar *thresPtr=thresIm.ptr<uchar>(y);
                uchar *labelsPtr=labels.ptr<uchar>(y);
                for(int x=0; x<thresIm.cols; x++)
                {
                    uchar reg = thresPtr[x];
                    uchar lleft_px = 0;
                    uchar ltop_px = 0;

                    if(x-1 > -1)
                    {
                        if(reg == thresPtr[x-1])
                            lleft_px =labelsPtr[x-1];
                    }

                    if(y-1 > -1)
                    {
                        if(reg ==thresIm.ptr<uchar>(y-1) [x]
                                )//thresIm.at<uchar>(y-1, x)
                            ltop_px =  labels.at<uchar>(y-1, x);
                    }

                    if(lleft_px==0 && ltop_px==0)
                        labelsPtr[x] = newLab++;

                    else if(lleft_px!=0 && ltop_px!=0)
                    {
                        if(lleft_px < ltop_px)
                        {
                            labelsPtr[x]  = lleft_px;
                            unions[ltop_px] = lleft_px;
                        }
                        else if(lleft_px > ltop_px)
                        {
                            labelsPtr[x]  = ltop_px;
                            unions[lleft_px] = ltop_px;
                        }
                        else
                        {//IGuales
                            labelsPtr[x]  = ltop_px;
                        }
                    }
                    else
                    {
                        if(lleft_px!=0) labelsPtr[x]  = lleft_px;
                        else labelsPtr[x]  = ltop_px;
                    }
                }
            }

            int nc= newLab-1 - unions.size();
            if(nc==2)
            {
                if(nZ > thresIm.total()-nZ) kp.class_id = 0;
                else kp.class_id = 1;
            }
            else if (nc > 2) {
                kp.class_id = 2;
            }
        }
    }
}
```

8. src/fractallabelers/fractalposetracker.h

```cpp

#include <opencv2/imgproc/imgproc.hpp>
#include "../picoflann.h"
#include "../cameraparameters.h"
#include "../markerdetector.h"
#include "fractalmarkerset.h"
#include "aruco_export.h"
namespace aruco {
    struct PicoFlann_KeyPointAdapter{
        inline  float operator( )(const cv::KeyPoint &elem, int dim)const { return dim==0?elem.pt.x:elem.pt.y; }
       inline  float operator( )(const cv::Point2f &elem, int dim)const { return dim==0?elem.x:elem.y; }
    };

    class ARUCO_EXPORT FractalPoseTracker
    {
    public:
        FractalPoseTracker();
        /**     init fractlPoseTracker parameters
         * @brief setParams
         * @param cam_params camera paremeters
         * @param msconf FractalMarkerSet configuration
         * @param markerSize FractalMarker size
         */
        void setParams(const CameraParameters& cam_params, const FractalMarkerSet& msconf, const float markerSize=-1);
        /**     estimate the pose of the fractal marker.
         * @brief fractalInnerPose
         * @param markerDetector
         * @param detected markers
         * @param refinement, use or not pose refinement. True by default.
         * @return true if the pose is estimated and false otherwise. If not estimated, the parameters m.Rvec and m.Tvec
         * and not set.
         */
        bool fractalInnerPose(const cv::Ptr<MarkerDetector> markerDetector, const std::vector<aruco::Marker>& markers, bool refinement=true);
        /**     extraction of the region of the image where the marker is estimated to be based on the previous pose
         * @brief ROI
         * @param imagePyramid set images
         * @param img original image. The image is scaled according to the selected pyramid image.
         * @param innerPoints2d collection fractal inner points. The points are scaled according to the selected pyramid image.
         * @param offset. Position of the upper inner corner of the marker. The offset is scaled according to the selected pyramid image.
         * @param ratio selected scaling factor
         */
        bool ROI(const std::vector<cv::Mat> imagePyramid, cv::Mat &img, std::vector<cv::Point2f> &innerPoints2d, cv::Point2f &offset, float &ratio);
        /**     classification of the corners of the marker
         * @brief assignClass
         * @param im image used in the classification
         * @param kpoints to classify
         * @param transf. Is it necessary to transform keypoints? (-MarkerSize/2 .. MarkerSize/2) to (0 .. ImageSize)
         * @param wsize. Window size
         */
        void assignClass(const cv::Mat &im, std::vector<cv::KeyPoint>& kpoints, bool transf=false, int wsize=5);

        /**     estimate the pose of the fractal marker. Method case 2, paper.
         * @brief fractal_solve_ransac
         * @param ninners (number of total inners points used)
         * @param inner_kpnt matches (inner points - detected keypoints)
         * @param kpnts keypoints
         * @param maxIter maximum number of iterations
         * @param _minInliers beta in paper
         * @param _thresInliers alpha in paper
         * @return Mat best model homography.
         */
        cv::Mat fractal_solve_ransac(int ninners, std::vector<std::pair<uint, std::vector<uint>>> inner_kpnt, std::vector<cv::KeyPoint> kpnts, uint32_t maxIter=500, float _minInliers=0.2f, float _thresInliers=0.7f);

        /**     Draw keypoints
         * @brief image
         * @param kpoints
         * @param text
         * @param transf. Is it necessary to transform keypoints? (-MarkerSize/2 .. MarkerSize/2) to (0 .. ImageSize)
         */
        void drawKeyPoints(const cv::Mat image, std::vector<cv::KeyPoint> kpoints, bool text=false, bool transf=false);

        /**     Refinement of the internal points of the marker and pose estimation with these.
         * @brief fractalRefinement
         * @param markerDetector
         * @param markerWarpPix. Optimal markerwarpPix used to select the image of the pyramid. Default value 10.
         * @return true if the pose is estimated and false otherwise. If not estimated, the parameters m.Rvec and m.Tvec
         * and not set.
         */
        bool fractalRefinement(const cv::Ptr<MarkerDetector> markerDetector, int markerWarpPix=10);

        // return the rotation vector. Returns an empty matrix if last call to estimatePose returned false
        const cv::Mat getRvec() const
        {
            return _rvec;
        }

        // return the translation vector. Returns an empty matrix if last call to estimatePose returned false
        const cv::Mat getTvec() const
        {
            return _tvec;
        }

        // return all corners from fractal marker
        const std::vector<cv::Point3f> getInner3d()
        {
            return _innerp3d;
        }

        //is the pose valid?
        bool isPoseValid()const{return !_rvec.empty() && !_tvec.empty();}

        FractalMarkerSet getFractal(){return _fractalMarker;}

    private:
        FractalMarkerSet _fractalMarker; //FractalMarkerSet configuration
        aruco::CameraParameters _cam_params; //Camera parameters.
        cv::Mat _rvec, _tvec;  // current poses
        std::map<int, std::vector<cv::Point3f>> _id_innerp3d; //Id_marker-Inners_corners
        std::vector<cv::Point3f> _innerp3d; //All inners corners
        std::vector<cv::KeyPoint> _innerkpoints; //All inners keypoints
        picoflann::KdTreeIndex<2,PicoFlann_KeyPointAdapter> _kdtree;
        std::map<int, double> _id_radius; //Idmarker_Radius(Optimus)
        std::map<int, float> _id_area; //Idmarker_projectedArea(Optimus)
        float _preRadius = 0; //radius used previous iteration
    };
}


```

9. src/CMakeLists.txt

```cpp

SET (LIBNAME ${EXTRALIBNAME}aruco)
include_directories(.)


SET(sources
    cameraparameters.cpp  debug.cpp             dictionary.cpp       ippe.cpp    markerdetector.cpp       markerlabeler.cpp  posetracker.cpp
    cvdrawingutils.cpp    dictionary_based.cpp   marker.cpp  markerdetector_impl.cpp  markermap.cpp  fractaldetector.cpp
    )
SET(headers
    aruco_cvversioning.h  cameraparameters.h  dictionary_based.h  ippe.h            markerdetector_impl.h  markermap.h    timers.h
    aruco_export.h        cvdrawingutils.h    dictionary.h        levmarq.h         marker.h               picoflann.h
    aruco.h               debug.h               markerdetector.h  markerlabeler.h        posetracker.h fractaldetector.h
    )
 set(fractal_sources
    fractallabelers/fractalposetracker.cpp
    fractallabelers/fractalmarkerset.cpp
    fractallabelers/fractalmarker.cpp
    fractallabelers/fractallabeler.cpp
    )
set(fractal_headers   
    fractallabelers/fractalposetracker.h
    fractallabelers/fractalmarkerset.h
    fractallabelers/fractalmarker.h
    fractallabelers/fractallabeler.h
    )

set(dcf_sources
    dcf/dcfmarkermaptracker.cpp  dcf/dcfmarkertracker.cpp  dcf/dcf_utils.cpp  dcf/trackerimpl.cpp    )
set(dcf_headers
    dcf/dcfmarkermaptracker.h  dcf/dcfmarkertracker.h  dcf/dcf_utils.h  dcf/trackerimpl.h)

add_library(${LIBNAME} ${sources} ${headers} ${fractal_sources} ${fractal_headers} ${dcf_sources} ${dcf_headers})
 
set_target_properties(${LIBNAME} PROPERTIES          # create *nix style library versions + symbolic links
    DEFINE_SYMBOL ARUCO_DSO_EXPORTS
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1                       # allow creating static and shared libs without conflicts
    OUTPUT_NAME "${LIBNAME}${PROJECT_DLLVERSION}"    # avoid conflicts between library and binary target names
)

target_link_libraries(${LIBNAME} PUBLIC opencv_core)
IF(BUILD_SVM)
add_definitions(USE_SVM_LABELER)
    target_link_libraries(${LIBNAME} PRIVATE opencv_imgproc opencv_calib3d opencv_features2d opencv_ml)
else()
    target_link_libraries(${LIBNAME} PRIVATE opencv_imgproc opencv_calib3d opencv_features2d )
endif()

INSTALL(TARGETS ${LIBNAME}
    RUNTIME DESTINATION bin COMPONENT main			# Install the dll file in bin directory
    LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT main
    ARCHIVE DESTINATION lib COMPONENT main)			# Install the dll.a file in lib directory

IF(ARUCO_DEVINSTALL)
   install(FILES ${headers}  DESTINATION include/aruco)
   install(FILES ${fractal_headers} DESTINATION include/aruco/fractallabelers)
   install(FILES ${dcf_headers} DESTINATION include/aruco/dcf)
ENDIF()
 
```

10. src/fractaldetector.cpp

```cpp

#include "fractaldetector.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "cvdrawingutils.h"
#include <algorithm>
#include "aruco_cvversioning.h"

namespace aruco
{
    FractalDetector::FractalDetector()
    {
        _markerDetector = new MarkerDetector();
    };

    void FractalDetector::setConfiguration(int params)
    {
         _fractalLabeler = FractalMarkerLabeler::create((FractalMarkerSet::CONF_TYPES)params);
         _params.configuration_type=FractalMarkerSet::getTypeString((FractalMarkerSet::CONF_TYPES)params);
         _markerDetector->setMarkerLabeler(_fractalLabeler);
    }

    void FractalDetector::setConfiguration(std::string params)
    {
        _params.configuration_type=params;
        _fractalLabeler = FractalMarkerLabeler::create(params);
        _markerDetector->setMarkerLabeler(_fractalLabeler);
    }

    void FractalDetector::drawMarkers(cv::Mat &img)
    {
        float size=  std::max(1.,float(img.cols)/ 1280.);
        for(auto m:Markers)
           m.draw(img, cv::Scalar(0, 0, 255), size, false);
    }

    void FractalDetector::draw2d(cv::Mat &img){
        if(Markers.size() > 0)
        {
            std::map<int, FractalMarker> id_fmarker = _fractalLabeler->_fractalMarkerSet.fractalMarkerCollection;

            std::vector<cv::Point2f> inners;
            std::map<int, std::vector<cv::Point3f>> id_innerCorners = _fractalLabeler->_fractalMarkerSet.getInnerCorners();
            for(auto id_innerC:id_innerCorners)
            {
                std::vector<cv::Point3f> inner3d;
                for(auto pt:id_innerC.second)
                    inners.push_back(cv::Point2f(pt.x,pt.y));
            }

            std::vector<cv::Point2f> points3d;
            std::vector<cv::Point2f> points2d;
            for(auto m:Markers)
            {
                for(auto p:id_fmarker[m.id].points)
                    points3d.push_back(cv::Point2f(p.x, p.y));

                for(auto p:m)
                    points2d.push_back(p);
            }

            cv::Mat H = cv::findHomography(points3d, points2d);
            std::vector<cv::Point2f> dstPnt;
            cv::perspectiveTransform(inners, dstPnt, H);

            float size=  std::max(1.,float(img.cols)/ 1280.);
            for(auto p:dstPnt)
                cv::circle(img, p, size, cv::Scalar(0,0,255), CV_FILLED);
        }
    }

    void FractalDetector::draw3d(cv::Mat &img, bool cube, bool axis){
        if(Tracker.isPoseValid())
        {
            cv::Mat rot;
            cv::Rodrigues(Tracker.getRvec(), rot);

            std::vector<cv::Point3f> innerPoints3d;
            for(auto pt:Tracker.getInner3d())
            {
                cv::Mat_<double> src(3,1,rot.type());
                src(0,0)=pt.x;src(1,0)=pt.y;src(2,0)=pt.z;

                cv::Mat  cam_image_point = rot * src + Tracker.getTvec();
                cam_image_point = cam_image_point/cv::norm(cam_image_point);

                if(cam_image_point.at<double>(2,0)>0.85)
                    innerPoints3d.push_back(pt);
            }
            //Draw inner points
            if(innerPoints3d.size() > 0)
            {
                std::vector<cv::Point2f> innerPoints;
                projectPoints(innerPoints3d, Tracker.getRvec(), Tracker.getTvec(), _cam_params.CameraMatrix, _cam_params.Distorsion, innerPoints);
                for(auto p:innerPoints)
                    circle(img, p, 3 ,  cv::Scalar(0,0,255),CV_FILLED);
            }
            //Draw cube
            if(cube)
            {
                std::map<int, FractalMarker> id_fmarker = Tracker.getFractal().fractalMarkerCollection;
                for(auto m:Markers)
                    draw3dCube(img, id_fmarker[m.id], _cam_params,  2);
            }

            //Draw axes
            if(axis)
                CvDrawingUtils::draw3dAxis(img, _cam_params, getRvec(), getTvec(), Tracker.getFractal().getFractalSize()/2);
        }
    }

    void FractalDetector::draw3dCube(cv::Mat& Image, FractalMarker m, const CameraParameters& CP, int lineSize)
    {
        cv::Mat objectPoints(8, 3, CV_32FC1);

        float msize= m.getMarkerSize();
        float halfSize = msize/2;

        objectPoints.at<float>(0, 0) = -halfSize;
        objectPoints.at<float>(0, 1) = -halfSize;
        objectPoints.at<float>(0, 2) = 0;
        objectPoints.at<float>(1, 0) = halfSize;
        objectPoints.at<float>(1, 1) = -halfSize;
        objectPoints.at<float>(1, 2) = 0;
        objectPoints.at<float>(2, 0) = halfSize;
        objectPoints.at<float>(2, 1) = halfSize;
        objectPoints.at<float>(2, 2) = 0;
        objectPoints.at<float>(3, 0) = -halfSize;
        objectPoints.at<float>(3, 1) = halfSize;
        objectPoints.at<float>(3, 2) = 0;

        objectPoints.at<float>(4, 0) = -halfSize;
        objectPoints.at<float>(4, 1) = -halfSize;
        objectPoints.at<float>(4, 2) = msize;
        objectPoints.at<float>(5, 0) = halfSize;
        objectPoints.at<float>(5, 1) = -halfSize;
        objectPoints.at<float>(5, 2) = msize;
        objectPoints.at<float>(6, 0) = halfSize;
        objectPoints.at<float>(6, 1) = halfSize;
        objectPoints.at<float>(6, 2) = msize;
        objectPoints.at<float>(7, 0) = -halfSize;
        objectPoints.at<float>(7, 1) = halfSize;
        objectPoints.at<float>(7, 2) = msize;


        std::vector<cv::Point2f> imagePoints;
        projectPoints(objectPoints, getRvec(), getTvec(), CP.CameraMatrix, CP.Distorsion, imagePoints);

        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i], imagePoints[(i + 1) % 4], cv::Scalar(0, 0, 255, 255), lineSize);

        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], cv::Scalar(0, 0, 255, 255), lineSize);

        for (int i = 0; i < 4; i++)
            cv::line(Image, imagePoints[i], imagePoints[i + 4], cv::Scalar(0, 0, 255, 255), lineSize);
    }
};
```

11. src/fractaldetector.h

```cpp

#ifndef _ARUCO_FractalDetector_H
#define _ARUCO_FractalDetector_H

#include "markerdetector.h"
#include "fractallabelers/fractallabeler.h"
#include "aruco_export.h"
namespace aruco {
    class ARUCO_EXPORT FractalDetector
    {
        struct ARUCO_EXPORT Params
        {
            std::string configuration_type;
        };

    public:
        FractalDetector();

        /**
         * @brief setConfiguration
         * @param configuration fractal id
         */
        void setConfiguration(int configuration);

        /**
         * @brief setConfiguration
         * @param configuration fractal file
         */
        void setConfiguration(std::string configuration);

        /**
         * @brief setParams
         * @param cam_params camera parameters
         * @param markerSize in meters
         */
        void setParams(const CameraParameters& cam_params, float markerSize)
        {
            _cam_params = cam_params;

            Tracker.setParams(cam_params, getConfiguration(), markerSize);
        }

        // return fractalmarkerset
        FractalMarkerSet getConfiguration()
        {
            return _fractalLabeler->_fractalMarkerSet;
        }

        // return true if any marker is detected, false otherwise
        bool detect(const cv::Mat& input)
        {
           Markers = _markerDetector->detect(input);

           if(Markers.size() > 0) return true;
           else return false;
        }

        // return true if the pose is estimated, false otherwise
        bool poseEstimation()
        {
            if (_cam_params.isValid())
            {
                return Tracker.fractalInnerPose(_markerDetector, Markers);
            }
            else
                return false;
        }

        // return the rotation vector. Returns an empty matrix if last call to estimatePose returned false
        cv::Mat getRvec(){
            return Tracker.getRvec();
        }
        // return the translation vector. Returns an empty matrix if last call to estimatePose returned false
        cv::Mat getTvec(){
            return Tracker.getTvec();
        }

        void drawImage(cv::Mat &img,cv::Mat &img2);

        // draw borders of markers
        void drawMarkers(cv::Mat &img);

        // draw inner corners of markers
        void draw2d(cv::Mat &img);

        // draw pose estimation axes
        void draw3d(cv::Mat &img, bool cube=true, bool axis=true);

        // draw marker as cube
        void draw3dCube(cv::Mat& Image, FractalMarker m, const CameraParameters& CP, int lineSize);

        // return detected markers
        std::vector<Marker> getMarkers()
        {
            return Markers;
        }

 private:
        // return image pyramid
        std::vector<cv::Mat> getImagePyramid()
        {
            return _markerDetector->getImagePyramid();
        }

        std::vector<aruco::Marker> Markers; //detected markers
        FractalPoseTracker Tracker;
        Params _params;
        CameraParameters _cam_params; //Camera parameters
        cv::Ptr<FractalMarkerLabeler> _fractalLabeler;
        cv::Ptr<MarkerDetector> _markerDetector;
    };
}
#endif


```

11. utils_fractal/CMakeLists.txt

```cpp

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
SET(THREADLIB "pthread")
ENDIF()

add_executable(fractal_create fractal_create.cpp)
add_executable(fractal_tracker fractal_tracker.cpp)
add_executable(fractal_print_marker fractal_print_marker.cpp)
add_executable(fractal_pix2meters fractal_pix2meters.cpp)

target_link_libraries(fractal_create		aruco opencv_calib3d opencv_highgui ${THREADLIB})
target_link_libraries(fractal_tracker		aruco opencv_calib3d opencv_highgui ${THREADLIB})
target_link_libraries(fractal_print_marker	aruco opencv_calib3d opencv_highgui ${THREADLIB})
target_link_libraries(fractal_pix2meters	aruco opencv_calib3d opencv_highgui ${THREADLIB})

INSTALL(TARGETS fractal_tracker RUNTIME DESTINATION bin)

```

12 . utils_fractal/fractal_create.cpp

```cpp

#include "fractallabelers/fractalmarkerset.h"
#include "dictionary.h"
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;

class CmdLineParser
{
    int argc;char** argv;public:
    CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}
    bool operator[](string param)
    {int idx = -1;   for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;return (idx != -1); }
    string operator()(string param, string defvalue = "-1"){int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param) idx = i;if (idx == -1) return defvalue;else return (argv[idx + 1]);}
};

std::vector<std::pair<int,int>> getRegionsConfig(string configuration){
    if (configuration.empty())return {};
    for(auto &c:configuration) if (c==',') c=' ';
    stringstream sstr(configuration);
    string markerConfig;
    std::vector<std::pair<int,int>> n_k;
    while(!sstr.eof()){
        if (sstr>>markerConfig)
        {
            int nVal, kVal;
            if (sscanf(markerConfig.c_str(), "%d:%d", &nVal, &kVal) != 2)
            {
                cerr << "Incorrect N:K specification" << endl;
                return {};
            }

            if(nVal <= kVal)
            {
                cerr << "Incorrect N:K specification. N should be > than K" << endl;
                return {};
            }

            n_k.push_back(make_pair(nVal,kVal));
        }
    }

    return n_k;
}

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 3 || cml["-h"])
        {
            cerr << "Usage: fractal_config.yml "
                    "n(f1):k(f1),n(f2):k(f2),...,n(fm),k(fm) "
                    "[-s bitSize (For the last level, in px. Default: -1, normalized marker)>]" << endl;

            cerr << endl;
            return -1;
        }

        //n(f1):k(f1),n(f2):k(f2),...,n(fm),k(fm)
        //Example fractal marker with 3 levels -> 10:8,14:10,6:0
        std::vector<std::pair<int,int>> regionsConfig;
        regionsConfig = getRegionsConfig(argv[2]);
        if(regionsConfig.size()<1) return -1;

        //bixSize (last level)
        int bitSize = stoi(cml("-s", "-1"));

        //Create configuration
        aruco::FractalMarkerSet fractalmarkerset;
        fractalmarkerset.create(regionsConfig, bitSize);

        //Save configuration file
        cv::FileStorage fs(argv[1], cv::FileStorage::WRITE);
        fractalmarkerset.saveToFile(fs);
    }
    catch (std::exception& ex)
    {
        cout << ex.what() << endl;
    }
}


```

14. utils_fractal/fractal_pix2meters.cpp

```cpp

// This program converts a boardconfiguration file expressed in pixel to another one expressed in meters
#include "fractallabelers/fractalmarkerset.h"
#include <iostream>

using namespace std;
using namespace aruco;
int main(int argc, char** argv)
{
    try
    {
        if (argc < 4)
        {
            cerr << "Usage: in_configuration.yml fractal_size(meters) out_configuration.yml" << endl;
            return -1;
        }

        FractalMarkerSet BInfo;
        BInfo = FractalMarkerSet::load(argv[1]);

        //Save file
        cv::FileStorage fs(argv[3], cv::FileStorage::WRITE);
        BInfo.convertToMeters(static_cast<float>(atof(argv[2]))).saveToFile(fs);
    }
    catch (std::exception& ex)
    {
        cout << ex.what() << endl;
    }
}


```

15. utils_fractal/fractal_print_marker.cpp

```cpp

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fractallabelers/fractalmarkerset.h>

using namespace cv;
using namespace std;

// convinience command line parser
class CmdLineParser
{
    int argc;
    char** argv;
public:
    CmdLineParser(int _argc, char** _argv)
          : argc(_argc)
          , argv(_argv)
    {
    }
    bool operator[](string param)
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        return (idx != -1);
    }
    string operator()(string param, string defvalue = "-1")
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        if (idx == -1)
            return defvalue;
        else
            return (argv[idx + 1]);
    }
};

int main(int argc, char** argv)
{
    try
    {        CmdLineParser cml(argc, argv);

        if (argc < 2)
        {
            cerr << "Usage: outfile.(jpg|png|ppm|bmp) [-c <configurationFile|CONF_TYPE>:FRACTAL_2L_6 default] [-bs:bitsize (smaller marker) 75 by default]"
                 << " [-noborder: removes the white border around the marker]" << endl;
            cerr << "\tConfigurations: ";
            for (auto config : aruco::FractalMarkerSet::getConfigurations())
                cerr << config << " ";
            return -1;
        }

        aruco::FractalMarkerSet fractalmarkerSet = aruco::FractalMarkerSet::load(cml("-c","FRACTAL_2L_6"));
        int pixSize = std::stoi(cml("-bs", "75"));  // pixel size each bit from smaller marker

        cv::Mat result = fractalmarkerSet.getFractalMarkerImage(pixSize,!cml["-noborder"]);
        cv::imwrite(argv[1], result);
    }
    catch (std::exception& ex)
    {
        cout << ex.what() << endl;
    }
}
```

16. utils_fractal/fractal_tracker.cpp

```cpp

#include "cvdrawingutils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

#include "fractaldetector.h"
#include "aruco_cvversioning.h"

using namespace std;
using namespace cv;
using namespace aruco;
struct   TimerAvrg{std::vector<double> times;size_t curr=0,n; std::chrono::high_resolution_clock::time_point begin,end;   TimerAvrg(int _n=30){n=_n;times.reserve(n);   }inline void start(){begin= std::chrono::high_resolution_clock::now();    }inline void stop(){end= std::chrono::high_resolution_clock::now();double duration=double(std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count())*1e-6;if ( times.size()<n) times.push_back(duration);else{ times[curr]=duration; curr++;if (curr>=times.size()) curr=0;}}double getAvrg(){double sum=0;for(auto t:times) sum+=t;return sum/double(times.size());}};
static TimerAvrg Tdetect, Tpose;

cv::Mat __resize(const cv::Mat& in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

// class for parsing command line
class CmdLineParser{int argc;char** argv;public:CmdLineParser(int _argc, char** _argv): argc(_argc), argv(_argv){}   bool operator[](string param)    {int idx = -1;  for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param)idx = i;return (idx != -1);}    string operator()(string param, string defvalue = "-1")    {int idx = -1;for (int i = 0; i < argc && idx == -1; i++)if (string(argv[i]) == param)idx = i;if (idx == -1)return defvalue;else return (argv[idx + 1]);}};

int main(int argc, char** argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 2 || cml["-h"])
        {
            cerr << "Usage: (video.avi|live[:index]) [-s fractalSize] [-cam cameraParams.yml] [-c <configuration.yml|CONFIG>:FRACTAL_2L_6 default]" << endl;
            cerr << "\tConfigurations: ";
            for (auto config : aruco::FractalMarkerSet::getConfigurations())
                cerr << config << " ";
            return 0;
        }

        aruco::CameraParameters CamParam;

        cv::Mat InImage;
        // Open input and read image
        VideoCapture vreader;
        bool isVideo=false;

        if(string(argv[1]).find("live")==std::string::npos){
            vreader.open(argv[1]);
             isVideo=true;
        }
        else{
            string livestr=argv[1];
            for(auto &c:livestr)if(c==':')c=' ';
            stringstream sstr;sstr<<livestr;
            string aux;int n=0;
            sstr>>aux>>n;
            vreader.open(n);
            if ( vreader.get(CV_CAP_PROP_FRAME_COUNT)>=2) isVideo=true;
        }


        if (vreader.isOpened())
            vreader >> InImage;
        else
        {
            cerr << "Could not open input" << endl;
            return -1;
        }

        // read camera parameters if passed
        if (cml["-cam"])
            CamParam.readFromXMLFile(cml("-cam"));

        // read marker size
        float MarkerSize = std::stof(cml("-s", "-1"));

        FractalDetector FDetector;
        FDetector.setConfiguration(cml("-c","FRACTAL_2L_6"));

        if (CamParam.isValid())
        {
            CamParam.resize(InImage.size());
            FDetector.setParams(CamParam, MarkerSize);
        }

        int frameId = 0;
        char key = 0;
        int waitTime=10;
        do
        {
            std::cout << "\r\rFrameId: " << frameId++<<std::endl;
            vreader.retrieve(InImage);

            // Ok, let's detect
            Tdetect.start();
            if(FDetector.detect(InImage))
            {
                std::cout << "Time detection: " << Tdetect.getAvrg()*1000 << " milliseconds"<<std::endl;
                Tdetect.stop();
                FDetector.drawMarkers(InImage);
            }

            //Pose estimation
            Tpose.start();
            if(FDetector.poseEstimation()){
                Tpose.stop();
                std::cout << "Time pose estimation: " << Tpose.getAvrg()*1000 << " milliseconds"<<std::endl;

                //Calc distance to marker
                cv::Mat tvec = FDetector.getTvec();
                double Z = sqrt(pow(tvec.at<double>(0,0),2) + pow(tvec.at<double>(1,0), 2) +
                         pow(tvec.at<double>(2,0),2));
                std::cout << "Distance to fractal marker: " << Z << " meters. "<<  std::endl;
                FDetector.draw3d(InImage); //3d
            }
            else
                FDetector.draw2d(InImage); //Ok, show me at least the inner corners!

            imshow("in", __resize(InImage, 1800));
            key = cv::waitKey(waitTime);  // wait for key to be pressed
            if (key == 's')
                waitTime = waitTime == 0 ? 10 : 0;

        } while (key != 27 && vreader.grab());
    }
    catch (std::exception& ex)
    {
        cout << "Exception :" << ex.what() << endl;
    }
}

```
