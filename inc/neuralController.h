/**
 * @file neural_controller.h
 * @author Jakob Schatzl
 * @brief
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef neuralController
#define neuralController

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOAD_WEIGHTS false

/**
 * @struct neuron_st
 * @brief Neuron for a feedforward network
 * @var netinput Calculated z of all connected outputs of the previous layer
 * @var netouput Calculated activation
 * @var bias Bias of the neuron that is added to all calculations of z
 * @var sigma Error signal needed for backpropagation
 */
typedef struct neuron {
    double netinput;
    double netoutput;
    double bias;
    double sigma;
} neuron_st;

/**
 * @struct arch_st
 * @brief Architecture of the neural network
 * @var topology                (Allocated) pointer to the topology of the neural network
 * @var total_neurons           Calculated value of all neurons, needed for sanity checks
 * @var total_weights           Calucalted value of all weights, needed for sanity checks
 */
typedef struct arch {
    int *topology;
    int total_neurons;
    int total_weights;
} arch_st;

/**
 * @struct neuralControllerConfig_st
 * @brief Configuration parameters for the neural network
 * @var hidden_layers           Number of hidden layers
 * @var layers                  Number of all layers
 * @var neurons                 Number of neurons per layer
 * @var output_layer_neurons    Number of neurons in the output layer
 * @var inputs                  Number of inputs into the first layer
 * @var max_epochs              (Optional) parameter in case maximum epochs variable needs to be saved in struct
 * @var initialized             Signals the API that the neural network finished initialing
 * @var learning_rate           Learning rate of the neural network
 * @var setpoint                Setpoint of the controller
 * @var isJordan                Boolean value to loop back the outputs to the inputs
 * @var arch                    Structure for the finished architecture of the network
 */
typedef struct neuralControllerConfig {
    int hidden_layers;
    int layers;
    int neurons;
    int output_layer_neurons;
    int inputs;
    int max_epochs;
    int initialized;
    double learning_rate;
    double setpoint;
    _Bool isJordan;
    arch_st arch;
} neuralControllerConfig_st;

/**
 * @struct control_st
 * @brief Structure needed to save important variables between runs
 * @var act_old                 Previously calcilated reinforcement learning variable
 * @var act_new                 Newly calculated reinforcement learning variable
 * @var rating                  Reinforcment learning variable that is injected into the backpropagation
 * @var input                   (Allocated) pointer to the inputs of the neural network
 * @var input_old               (Allocated) pointer to the previous inputs of the neural network
 * @var epoch                   Current epoch
 */
typedef struct control {
    double act_old;
    double act_new;
    double rating;
    double *input;
    double *input_old;
    unsigned int epoch;
} control_st;

/**
 * @struct input_st
 * @brief Thread-safe variant for input variables with availability checks. Currently not needed
 */
typedef struct input {
    double value;
    _Bool available;
} input_st;

/**
 * @brief Initializes the neural controller
 * @details Parses the neuralControllConfig_st config and calculates the topology(layers + hidden layers) of the network, as well as
 *          the toal weights and neurons (these are used solely for debugging purposes). Then it allocates memory on the basis of the
 *          topology array and finally initializes the values of the weights and biases using the fctPtr function, which ideally should
 *          be a function generating values between 0 and 1.
 * @param ncConfig Reference pointer to a neuralControllerConfi_st structure
 * @param control Reference pointer to a control_st structure
 * @param fctPtr Pointer to a function that generates values between 0 and 1
 * @param pWeight Reference pointer to a 3-dimensional weights array
 * @param pneuron Reference pointer to a 2-dimensional neuron_st array
 * @return 0 on success
 */
int neuralController_Init(neuralControllerConfig_st* ncConfig, control_st *control, float (*fctPtr)(), double**** pWeight, neuron_st*** pNeuron);

/**
 * @brief Iterates once through the forward and backwards pass.
 * @details Parses the input pointer, caluclates a forward pass, then caluclates values used for reinforcement learning and controlling parameters, 
 *          which are then written into the pOutput pointer, and finally calulates the backwards pass.
 * @param ncConfig Reference pointer to a neuralControllerConfi_st structure
 * @param control Reference pointer to a control_st structure
 * @param pOutput Reference pointer to the output
 * @param pInput Reference pointer to array of inputs outside of what is calculated in the function
 * @param weight Pointer to a 3-dimensional weights array
 * @param neuron Pointer to a 2-dimensional neuron_st array
 * @return 0 on success
 */
int neuralController_Run(neuralControllerConfig_st* ncConfig, control_st *control, double* pOutput, double* pInput, double*** weight, neuron_st** neuron);

/**
 * @brief Frees the allocated memory used for the neural controller.
 * @details Frees the allocated memory of the weights array, the neuron array, the input array, the input_old array and the topology array.
 * @param ncConfig The neuralControllerConfig_st structure used for iteration in the for loops
 * @param control The structure for the input and input_old arrays
 * @param weight Pointer to a 3-dimensional weights array
 * @param neuron Pointer to a 2-dimensional neuron_st array
 */
void neuralController_Free(neuralControllerConfig_st* ncConfig, control_st *control, double ***weight, neuron_st **neuron);

/**
 * @brief Currently unused. Please don't use
 */
void saveArrayToFile(const char *filename);

/**
 * @brief C function for the hyberbolic tangent
 * @param x x value for the dervative of the hyberbolic tangent
 * @return y value for the dervative of the hyberbolic tangent
 */
double dTanh(double x);

/**
 * @brief Sigmoid function
 * @param x x value
 * @return y value
 */
double sigmoid(double x);

/**
 * @brief Derivative of the sigmoid function
 * @param x x value
 * @return y value
 */
double dSigmoid(double x);

#endif  /* neuralController */
