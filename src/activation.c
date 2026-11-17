// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file activation.c
 * @author Jakob Schatzl
 * @brief Implementation of activation functions for the neural controller
 * @version 1.0
 * @date 2026-10-05
 */
/**************************************************************************************************
                                    Section for headers
***************************************************************************************************/
#include "activation.h"
#include "neuralController.h"

/**************************************************************************************************
                        Section for public function prototypes
***************************************************************************************************/
/**
 * @brief Sigmoid function
 * @param x x value
 * @return y value
 */
static float sigmoid(float x);

/**
 * @brief Derivative of the sigmoid function
 * @param x x value of the sigmoid function
 * @return y value of the sigmoid function
 */
static float dSigmoid(float x);

/**
 * @brief Derivative of the hyberbolic tangent
 * @param x x value of the dervative of the hyberbolic tangent
 * @return y value of the dervative of the hyberbolic tangent
 */
static float dtanhf(float x);

/**
 * @brief Leaky ReLU function
 * @param x x value of the leaky ReLU function
 * @return y value of the leaky ReLU function
 */
static float leakyRelu(float x);

/**
 * @brief Derivative of the leaky ReLU function
 * @param x x value for the dervative of the leaky ReLU function
 * @return y value for the dervative of the leaky ReLU function
 */
static float dLeakyRelu(float x);

/**
 * @brief Linear function
 * @param x x value of the linear function
 * @return y value of the linear function
 */
static float linear(float x);

/**
 * @brief Derivative of the linear function
 * @param x x value for the dervative of the linear function (unused but required for consistency)
 * @return y value for the dervative of the linear function
 */
static float dLinear(float x);

/**************************************************************************************************
                                Section for static functions
***************************************************************************************************/
static float sigmoid(float x) {
    return 1 / (1 + expf(-x)); 
}

static float dSigmoid(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

static float dtanhf(float x) {
    float th = tanhf(x);
    return 1.0 - th * th;
}

static float leakyRelu(float x) {
    static const float alpha = 0.01;
    return x > 0.0 ? x : alpha * x;
}

static float dLeakyRelu(float x) {
    static const float alpha = 0.01;
    return x > 0.0 ? 1.0 : alpha;
}

static float linear(float x) {
    return x;
}

static float dLinear(float x) {
    /* For consistency only, as the function pointer will still need a parameter -> tell the compiler x is not used */
    (void)x;
    return 1.0;
}

/**************************************************************************************************
                                Section for public functions
***************************************************************************************************/


/**************************************************************************************************
                                    Constant instances
***************************************************************************************************/
const activation_st ACT_SIGMOID = {
    .forward = sigmoid,
    .backward = dSigmoid,
};

const activation_st ACT_tanhf = {
    .forward = tanhf,
    .backward = dtanhf,
};

const activation_st ACT_LEAKY_RELU = {
    .forward = leakyRelu,
    .backward = dLeakyRelu,
};

const activation_st ACT_LINEAR = {
    .forward = linear,
    .backward = dLinear,
};