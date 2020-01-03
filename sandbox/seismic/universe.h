#include <chrono>
#include <string>
#include <cstring>

#ifndef UNIVERSE_H_
#define UNIVERSE_H_

inline const int UNIVERSE_WIDTH {1024}; 
inline const int UNIVERSE_HEIGHT {512};

#include "video.h"

class Universe {
public:
    enum {
        UniverseWidth  = UNIVERSE_WIDTH,
        UniverseHeight = UNIVERSE_HEIGHT
    };
private:
    //in order to avoid performance degradation due to cache aliasing issue
    //some padding is needed after each row in array, and between array themselves.
    //the padding is achieved by adjusting number of rows and columns.
    //as the compiler is forced to place class members of the same clause in order of the
    //declaration this seems to be the right way of padding.

    //magic constants added below are chosen experimentally for 1024x512.
    enum {
        MaxWidth = UniverseWidth+1,
        MaxHeight = UniverseHeight+3
    };

    typedef float ValueType;

    //! Horizontal stress
    ValueType S[MaxHeight][MaxWidth];

    //! Velocity at each grid point
    ValueType V[MaxHeight][MaxWidth];

    //! Vertical stress
    ValueType T[MaxHeight][MaxWidth];

    //! Coefficient related to modulus
    ValueType M[MaxHeight][MaxWidth];

    //! Damping coefficients
    ValueType D[MaxHeight][MaxWidth];

    //! Coefficient related to lightness
    ValueType L[MaxHeight][MaxWidth];

    enum { ColorMapSize = 1024};
    color_t ColorMap[4][ColorMapSize];

    enum MaterialType {
        WATER=0,
        SANDSTONE=1,
        SHALE=2
    };

    //! Values are MaterialType, cast to an unsigned char to save space.
    unsigned char material[MaxHeight][MaxWidth];

private:
    enum { DamperSize = 32};

    int pulseTime;
    int pulseCounter;
    int pulseX;
    int pulseY;

    drawing_memory drawingMemory;

    std::string _model {"tf"};

public:
    const std::string& get_model() { return _model; }
    void set_model(std::string m) { _model = m; }
    void InitializeUniverse(video const& colorizer);

    bool TryPutNewPulseSource(int x, int y);
    void SetDrawingMemory(const drawing_memory &dmem);

    struct Rectangle {
      struct std::pair<int,int> xRange;
      struct std::pair<int,int> yRange;
      Rectangle (int startX, int startY, int width, int height):xRange(startX,width),yRange(startY,height){}
      int StartX() const {return xRange.first;}
      int StartY() const {return yRange.first;}
      int Width()   const {return xRange.second;}
      int Height()  const {return yRange.second;}
      int EndX() const {return xRange.first + xRange.second;}
      int EndY() const {return yRange.first + yRange.second;}
    };

    void UpdatePulse();
    void UpdateStress(Rectangle const& r);
    void UpdateVelocity(Rectangle const& r);

    friend struct UpdateStressBody;
    friend struct UpdateVelocityBody;

    //void ParallelUpdateUniverse();
    //void ParallelUpdateStress(tbb::affinity_partitioner &affinity);
    //void ParallelUpdateVelocity(tbb::affinity_partitioner &affinity);

    void SerialUpdateVelocity();
    void SerialUpdateUniverse();
    void SerialUpdateStress();
};

std::chrono::microseconds measure_time_tbb(unsigned, unsigned, Universe &);
std::chrono::microseconds measure_time_taskflow(unsigned, unsigned, Universe &);


#endif /* UNIVERSE_H_ */
