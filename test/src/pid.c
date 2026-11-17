// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pid.c
 * @author Jakob Schatzl
 * @brief Implementation of the PID controller
 * @version 1.0
 * @date 2026-11-17
 */
#include "pid.h"

void PID_Init(PID_Handle_t *pid,
              float kp,
              float ki,
              float kd,
              float ts,
              float out_min,
              float out_max)
{
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->ts = ts;

    pid->out_min = out_min;
    pid->out_max = out_max;

    pid->integral = 0.0f;
    pid->measurement_prev = 0.0f;
    pid->output_prev = 0.0f;
    pid->initialized = 0u;
}

void PID_Reset(PID_Handle_t *pid)
{
    pid->integral = 0.0f;
    pid->measurement_prev = 0.0f;
    pid->output_prev = 0.0f;
    pid->initialized = 0u;
}

float PID_Update(PID_Handle_t *pid,
                 float setpoint,
                 float measurement)
{
    float error = setpoint - measurement;

    /* Derivative on measurement avoids derivative kick on setpoint steps. */
    float derivative = 0.0f;

    if (pid->initialized != 0u) {
        derivative = (pid->measurement_prev - measurement) / pid->ts;
    } else {
        pid->measurement_prev = measurement;
        pid->initialized = 1u;
    }

    /* Tentative integral update. */
    float integral_new =
        pid->integral + pid->ki * pid->ts * error;

    /* Unsaturated PID output. */
    float output_unsat =
        pid->kp * error +
        integral_new +
        pid->kd * derivative;

    /* Saturate plant/DAC command. */
    float output_sat = output_unsat;

    if (output_sat < pid->out_min) {
        output_sat = pid->out_min;
    } else if (output_sat > pid->out_max) {
        output_sat = pid->out_max;
    }

    /*
     * Conditional anti-windup:
     * integrate only when doing so does not push an already-saturated
     * output farther into saturation.
     */
    if ((output_unsat == output_sat) ||
        ((error > 0.0f) && (output_sat < pid->out_max)) ||
        ((error < 0.0f) && (output_sat > pid->out_min))) {
        pid->integral = integral_new;
    }

    pid->measurement_prev = measurement;
    pid->output_prev = output_sat;

    return output_sat;
}