# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file pt2_rk4.py
# @author Jakob Schatzl
# @brief PT2 system simulation using the Runge-Kutta 4th order method
# @version 1.0
# @date 2026-11-17

class PT2:
    def __init__(self, K: float, T: float, D: float, dt: float):
        self.K = K
        self.T = T
        self.D = D
        self.dt = dt
        self.y = 0.0
        self.dy = 0.0

    def _derivative(self, y, dy, u):
        ddy = (self.K * u - 2.0 * self.D * self.T * dy - y) / (self.T ** 2)
        return dy, ddy

    def step(self, u: float) -> float:
        dt = self.dt

        k1_y, k1_dy = self._derivative(self.y, self.dy, u)

        y2 = self.y + 0.5 * dt * k1_y
        dy2 = self.dy + 0.5 * dt * k1_dy
        k2_y, k2_dy = self._derivative(y2, dy2, u)

        y3 = self.y + 0.5 * dt * k2_y
        dy3 = self.dy + 0.5 * dt * k2_dy
        k3_y, k3_dy = self._derivative(y3, dy3, u)

        y4 = self.y + dt * k3_y
        dy4 = self.dy + dt * k3_dy
        k4_y, k4_dy = self._derivative(y4, dy4, u)

        self.y += (dt / 6.0) * (
            k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y
        )

        self.dy += (dt / 6.0) * (
            k1_dy + 2.0 * k2_dy + 2.0 * k3_dy + k4_dy
        )

        return self.y