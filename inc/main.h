// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.h
 * @author Jakob Schatzl
 * @brief Public header for the main program
 * @version 1.0
 * @date 2026-11-17
 */
#ifndef MAIN_H
#define MAIN_H

#include <errno.h>
#include <time.h>
//#include <zmq.h>

#include "neuralController.h"
#include "activation.h"

#define PlotGraph false
#define WriteToFile false

typedef struct i2 {
    float yn;
    float yn2;
} i2_st;

float generateRandomInt(void);
float i_plant(float yn, float u);
float round_to_decimal_places(float value, int decimal_places);

static const char inifile[] = "F:/work/Neuro-Controller/cfg/config.ini";

#endif  // MAIN_H