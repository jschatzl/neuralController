#ifndef main_h
#define main_h

#include <errno.h>
#include <time.h>
//#include <zmq.h>

#include "neuralController.h"

#define PlotGraph false
#define WriteToFile false

typedef struct i2 {
    float yn;
    float yn2;
} i2_st;

float generateRandomInt(void);
float i_plant(float yn, float u);
double round_to_decimal_places(float value, int decimal_places);

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

#endif  // main_h