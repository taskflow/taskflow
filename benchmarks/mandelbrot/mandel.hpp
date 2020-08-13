#include <cmath>
#include <string>
#include <vector>
#include <tuple>
#include <chrono>
#include <cassert>

const int D = 2;
const int MAX_ITERATION = 200;
const double XL = - 2.5;
const double XR = 1.0;
const double YB = -1.0;
const double YT = 1.0;
extern int H; 
extern int W; 
extern unsigned char* RGB;


inline void dump_tga(int w, int h, unsigned char rgb[], const char *filename) {
  FILE *file_unit;
  unsigned char header1[12] = {0,0,2,0,0,0,0,0,0,0,0,0};
  unsigned char header2[6] = {static_cast<unsigned char>(w%256), 
                              static_cast<unsigned char>(w/256), 
                              static_cast<unsigned char>(h%256), 
                              static_cast<unsigned char>(h/256), 24, 0};

  // Create the file.
  file_unit = fopen(filename, "wb");

  // Write the headers.
  fwrite(header1, sizeof(unsigned char), 12, file_unit);
  fwrite(header2, sizeof(unsigned char), 6, file_unit);

  //Write the image data.
  fwrite(rgb, sizeof(unsigned char), 3*w*h, file_unit);

  // Close the file.
  fclose(file_unit);

  printf("Dump figure to '%s'\n", filename);
}


// Computes the value of the point (px, py), which represents the complex
// number (px + i*py), based on whether it is part of the Mandelbrot set or not
inline int escape_time(double px, double py, int n) {
  // x and y represent a complex number (x + i*y)
  double x = px;
  double y = py;

  // square both components
  double x2 = x * x;
  double y2 = y * y; 

  //// Julia set
  //double cr = -0.8;
  //double ci = 0.156;

  // Mandelbrot set: 
  double cr = px;
  double ci = py;

  double xtmp;
  
  // We need i (number of iteration) after the loop, so don't declare the variable inside the for.
  int i;
  for (i = 0; (x2 + y2) <= 4.0 && i < MAX_ITERATION; i++) { 
    xtmp = pow((x*x+y*y), (n/2.0))*cos(n*atan2(y,x)) + cr;
    y = pow((x*x+y*y), (n/2.0))*sin(n*atan2(y,x)) + ci;
    x = xtmp;

    // Update our temp variables
    x2 = x * x;
    y2 = y * y;
  }

  return i;
}

// Map a given (x, y) coordinate to the range [XL, XR] [YB, YT]
inline std::pair<double, double> scale_xy(double x, double y) {
  double xx = XL + (XR - XL)/W * x;
  double yy = YB + (YT - YB)/H * y;
  return {xx, yy};
} 

// Given an iteration, return a RGB color 
// Ref: https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia
inline std::tuple<int, int, int> get_color(int n) {
  if(n < MAX_ITERATION && n) {
    int i = n % 16;
    const static std::vector<std::tuple<int, int, int>> colors {
      {66, 30, 15},
      {25, 7, 26},
      {9, 1, 47},
      {4, 4, 73},
      {0, 7, 100},
      {12, 44, 138},
      {24, 82, 177},
      {57, 125, 209},
      {134, 181, 229},
      {211, 236, 248},
      {241, 233, 191},
      {248, 201, 95},
      {255, 170, 0},
      {204, 128, 0},
      {153, 87, 0},
      {106, 52, 3}
    };
    return colors[i];
  }
  return {0, 0, 0};
}


std::chrono::microseconds measure_time_taskflow(unsigned);
std::chrono::microseconds measure_time_omp(unsigned);
std::chrono::microseconds measure_time_tbb(unsigned);
