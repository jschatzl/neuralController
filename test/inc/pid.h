// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pid.h
 * @author Jakob Schatzl
 * @brief Public header for the PID controller implementation
 * @version 1.0
 * @date 2026-11-17
 */
#ifndef PID_H
#define PID_H

typedef struct PID_Handle {
    float kp;
    float ki;
    float kd;
    float ts;

    float out_min;
    float out_max;

    float integral;
    float measurement_prev;
    float output_prev;

    unsigned int initialized;
} PID_Handle_t;

void PID_Init(PID_Handle_t *pid,
              float kp,
              float ki,
              float kd,
              float ts,
              float out_min,
              float out_max);

float PID_Update(PID_Handle_t *pid,
                 float setpoint,
                 float measurement);

void PID_Reset(PID_Handle_t *pid);

#endif /* PID_H */