// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pt2.c
 * @author Jakob Schatzl
 * @brief Implementation of the PT2 controller
 * @version 1.0
 * @date 2026-11-17
 */
#include "pt2.h"

typedef struct {
    float y;
    float dy;
} PT2_State_t;

static PT2_State_t PT2_Derivative(const PT2_Handle_t *h,
                                  PT2_State_t x,
                                  float u)
{
    PT2_State_t dx;
    dx.y = x.dy;

    dx.dy = (h->K * u
           - (2.0f * h->D * h->T * x.dy)
           - x.y)
          / (h->T * h->T);

    return dx;
}

void PT2_Init(PT2_Handle_t *h, float Ts)
{
    h->K  = 1.0f;
    h->T  = 0.5f;
    h->D  = 1.0f;
    h->Ts = Ts;

    h->y  = 0.0f;
    h->dy = 0.0f;
}

void PT2_Reset(PT2_Handle_t *h, float y0, float dy0)
{
    h->y  = y0;
    h->dy = dy0;
}

float PT2_Update(PT2_Handle_t *h, float u)
{
    const float dt = h->Ts;

    PT2_State_t x = {
        .y  = h->y,
        .dy = h->dy
    };

    PT2_State_t k1 = PT2_Derivative(h, x, u);

    PT2_State_t x2 = {
        .y  = x.y  + 0.5f * dt * k1.y,
        .dy = x.dy + 0.5f * dt * k1.dy
    };
    PT2_State_t k2 = PT2_Derivative(h, x2, u);

    PT2_State_t x3 = {
        .y  = x.y  + 0.5f * dt * k2.y,
        .dy = x.dy + 0.5f * dt * k2.dy
    };
    PT2_State_t k3 = PT2_Derivative(h, x3, u);

    PT2_State_t x4 = {
        .y  = x.y  + dt * k3.y,
        .dy = x.dy + dt * k3.dy
    };
    PT2_State_t k4 = PT2_Derivative(h, x4, u);

    h->y += (dt / 6.0f) *
            (k1.y + 2.0f * k2.y + 2.0f * k3.y + k4.y);

    h->dy += (dt / 6.0f) *
             (k1.dy + 2.0f * k2.dy + 2.0f * k3.dy + k4.dy);

    return h->y;
}