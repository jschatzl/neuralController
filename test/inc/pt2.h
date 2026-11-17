// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pt2.h
 * @author Jakob Schatzl
 * @brief Public header for the PT2 controller implementation
 * @version 1.0
 * @date 2026-11-17
 */
#ifndef PT2_H
#define PT2_H

typedef struct PT2_Handle {
    float K;      // Static gain
    float T;      // Time constant [s]
    float D;      // Damping ratio
    float Ts;     // Simulation/control sampling time [s]

    float y;      // Plant output
    float dy;     // First derivative of y
} PT2_Handle_t;

void PT2_Init(PT2_Handle_t *h, float Ts);
void PT2_Reset(PT2_Handle_t *h, float y0, float dy0);
float PT2_Update(PT2_Handle_t *h, float u);

#endif /* PT2_H */