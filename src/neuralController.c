// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file neural_controller.c
 * @author Jakob Schatzl
 * @brief Implementation of library functions for the neural controller
 * @version 1.0
 * @date 2026-11-17
 */
/**************************************************************************************************
                                    Section for headers
***************************************************************************************************/
#include "neuralController.h"
#include "activation.h"

/**************************************************************************************************
                            Section for private function prototypes
***************************************************************************************************/
/**
 * @brief Calculates the whole network
 * @details Function to calculate all neurons in all layers, first the z with bias and then the activation function
 * @param ncConfig The neuralControllerConfig_st structure used for iteration in the for loops
 * @param control The structure for the input and input_old arrays
 * @param weight Pointer to a 3-dimensional weights array
 * @param neuron Pointer to a 2-dimensional neuron_st array
 */
static void forwardPass(neuralControllerConfig_st* ncConfig, control_st *control, float ***weight, neuron_st **neuron);

/**
 * @brief Calculates the sigma values needed for \ref updateWeightsAndBiases() 
 * @details Calulcates the sigma values starting from the output through the hidden layers to the input layers. Afterwards,
 *          \ref updateWeightsAndBiases() shall be called to update the weights and biases with the newly calulcated sigma values.
 * @param ncConfig The neuralControllerConfig_st structure used for iteration in the for loops
 * @param control The structure for the input and input_old arrays
 * @param weight Pointer to a 3-dimensional weights array
 * @param neuron Pointer to a 2-dimensional neuron_st array
 */
void computeSigmas(neuralControllerConfig_st* ncConfig, float* pInput, float ***weight, neuron_st **neuron);

/**
 * @brief Updates the weights and biases
 * @details Updates the weights and biases on the basis of the calculated sigmas in \ref computeSigmas().
 * @param ncConfig The neuralControllerConfig_st structure used for iteration in the for loops
 * @param control The structure for the input and input_old arrays
 * @param weight Pointer to a 3-dimensional weights array
 * @param neuron Pointer to a 2-dimensional neuron_st array
 */
static void updateWeightsAndBiases(neuralControllerConfig_st* ncConfig, control_st *control, float ***weight, neuron_st **neuron);

/**
 * @brief Fetches the pointer to the cativation instances
 * @param id The type of the activation in \ref activation_type_en
 * @return The activation instance
 */
static const activation_st *get_activation(activation_type_en id);

static inline float clampf(float value, float min, float max);

/**************************************************************************************************
                                Section for public functions
***************************************************************************************************/
int neuralController_Init(neuralControllerConfig_st* ncConfig, control_st *control, float (*fctPtr)(float, float), float**** pWeight, neuron_st*** pNeuron) {
    /** Null pointer check */
    if((!ncConfig) || (!control) || (!fctPtr) || (!pWeight) || (!pNeuron)){
        return -1;
    }
    /** neuralControllerConfig_st values check */
    if((ncConfig->inputs == 0) || (ncConfig->hidden_layers == 0) || (ncConfig->layers == 0) || (ncConfig->neurons == 0) || (ncConfig->output_layer_neurons == 0)){
        return -1;
    }

    /** Check if activation function pointers exist */
    if((ncConfig->activation_hidden == 0) || (ncConfig->activation_output == 0)){
        return -1;
    }

    control->act_new = 0;
    control->act_old = ncConfig->setpoint - 0;
    if(ncConfig->isJordan){
        ncConfig->inputs += ncConfig->output_layer_neurons;
    }
    control->input = (float*)calloc(ncConfig->inputs, sizeof(float));
    control->input_old = (float*)calloc(ncConfig->inputs, sizeof(float));
    control->rating = 0;

    ncConfig->arch.total_neurons = ncConfig->neurons * ncConfig->hidden_layers + ncConfig->output_layer_neurons;
    ncConfig->arch.total_weights = (ncConfig->inputs * ncConfig->neurons) + (ncConfig->neurons * ncConfig->neurons * (ncConfig->hidden_layers - 1)) + (ncConfig->neurons * ncConfig->output_layer_neurons);

    const activation_st* act_hidden = get_activation(ncConfig->activation_hidden);
    const activation_st* act_output = get_activation(ncConfig->activation_output);

    ncConfig->arch.topology = (layer_st *)calloc(ncConfig->layers, sizeof(layer_st));
    for (int i = 0; i < ncConfig->layers; i++) {
        if (i == ncConfig->layers - 1) {
            /** Output layer */
            ncConfig->arch.topology[i].neurons = ncConfig->output_layer_neurons;
            ncConfig->arch.topology[i].activation = act_output;
        } else if (i == 0) {
            /** Input layer, no activation */
            ncConfig->arch.topology[i].neurons = ncConfig->inputs;
            ncConfig->arch.topology[i].activation = act_hidden;
        } else {
            /** Hidden layer */
            ncConfig->arch.topology[i].neurons = ncConfig->neurons;
            ncConfig->arch.topology[i].activation = act_hidden;
        }
    }

#if LOAD_weight

    void loadArrayFromFile(const char *filename) {
        FILE *file = fopen(filename, "rb");
        if (file == NULL) {
            perror("Error opening file");
            exit(EXIT_FAILURE);
        }

        // Read the entire 3D array from the file
        size_t elements = total_weight;
        fread(weight, sizeof(float), total_weight, file);
        fclose(file);
    }

    /*Initialize weight and bias with values from the .bin and
      initialize the rest with 0*/
    for (int layer = 0; layer < ncConfig->layers; layer++) {
        for (int j = 0; j < ncConfig->arch.topology[layer]; j++) {
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = 0;
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }
    ncConfig->initialized = 1;

#else /*!LOAD_weight*/

    /*Initialize weight and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    float ***weight = (float ***)calloc(ncConfig->layers - 1, sizeof(float **));
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        weight[layer] = (float **)calloc(ncConfig->arch.topology[layer].neurons, sizeof(float *));
        for (int j = 0; j < ncConfig->arch.topology[layer].neurons; j++) {
            weight[layer][j] = (float *)calloc(ncConfig->arch.topology[layer + 1].neurons, sizeof(float));
            for (int k = 0; k < ncConfig->arch.topology[layer + 1].neurons; k++) {
                weight[layer][j][k] = (float)(*fctPtr)(ncConfig->arch.topology[layer].neurons, ncConfig->arch.topology[layer+1].neurons);
            }
        }
    }

    neuron_st **neuron = (neuron_st **)calloc(ncConfig->layers, sizeof(neuron_st *));
    for (int layer = 1; layer < ncConfig->layers; layer++) {
        neuron[layer - 1] = (neuron_st *)calloc(ncConfig->arch.topology[layer].neurons, sizeof(neuron_st));
        for (int j = 0; j < ncConfig->arch.topology[layer].neurons; j++) {
            /*Initialize bias and everything else in the neuron struct*/
            if(layer == ncConfig->layers-1){
                neuron[layer - 1][j].bias = 0.5;
            }else{
                neuron[layer - 1][j].bias = 0.0;
            }
            neuron[layer - 1][j].netinput = 0.0;
            neuron[layer - 1][j].netoutput = 0.0;
            neuron[layer - 1][j].sigma = 0.0;
        }
    }
    *pWeight = weight;
    *pNeuron = neuron;
    ncConfig->initialized = 1;
    
#endif /*LOAD_weight*/

    return 0;
}

int neuralController_Run(neuralControllerConfig_st* ncConfig, control_st *control, float* pOutput, float* pInput, float*** weight, neuron_st** neuron) {
    /** Null pointer check */
    if((!ncConfig) || (!control) || (!pOutput) || (!pInput) || (!weight) || (!neuron)){
        return -1;
    }
    control->input[0] = ncConfig->setpoint - pInput[0];
    control->input[1] = pInput[0];

    /* Loop back output as input for Jordan type network */
    if(ncConfig->isJordan){
        control->input[2] = *pOutput;
    }
    forwardPass(ncConfig, control, weight, neuron);
    computeSigmas(ncConfig, pInput, weight, neuron);
    updateWeightsAndBiases(ncConfig, control, weight, neuron);

    control->epoch++;

    *pOutput = neuron[ncConfig->hidden_layers][0].netoutput;
#if LOG_ENABLE
    if((control->epoch % 1000) == 0){
         printf("Setpoint: %f, Learning rate: %f, Plant output: %f, u: %f,  Error: %f \n", ncConfig->setpoint, ncConfig->learning_rate, pInput[1], *pOutput, ncConfig->setpoint - pInput[0]);
    }
#endif
    return 0;
}

void neuralController_Free(neuralControllerConfig_st* ncConfig, control_st *control, float ***weight, neuron_st **neuron) {
    if((!weight) || (!neuron) || (!ncConfig))
        return;

    for(int layer = 0; layer < ncConfig->layers - 1; layer++){
        for(int j = 0; j < ncConfig->arch.topology[layer].neurons; j++){
            free(weight[layer][j]);
        }
        free(weight[layer]);
    }
    
    for(int layer = 0; layer < ncConfig->layers - 1; layer++){
        free(neuron[layer]);
    }

    free(control->input);
    free(control->input_old);
    free(ncConfig->arch.topology);
    free(weight);
    free(neuron);
}

/**************************************************************************************************
                                Section for private functions
***************************************************************************************************/

static void forwardPass(neuralControllerConfig_st* ncConfig, control_st *control, float ***weight, neuron_st **neuron){
    int n = 0; /* Sanity check variable to count neurons during forward pass */

    /* Feed forward network */
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        for (int j = 0; j < ncConfig->arch.topology[layer + 1].neurons; j++) {
            float sum = neuron[layer][j].bias;
            for (int k = 0; k < ncConfig->arch.topology[layer].neurons; k++) {
                if (layer == 0)
                    sum += control->input[k] * weight[layer][k][j];
                else
                    sum += neuron[layer - 1][k].netoutput * weight[layer][k][j];
            }
            neuron[layer][j].netinput = sum;
            neuron[layer][j].netoutput = ncConfig->arch.topology[layer+1].activation->forward(sum);
            n++;
        }
    }
    /* Sanity check to see if all neuron values have been calculated */
    assert(n == ncConfig->arch.total_neurons);
}

void computeSigmas(neuralControllerConfig_st* ncConfig, float* pInput, float ***weight, neuron_st **neuron){
    int n = 0; /* Sanity check variable to count neurons during sigma calculation */

    /*Backpropagation*/
    /*For detailed expflaination see https://en.wikipedia.org/wiki/Backpropagation
     * next layer     = k = layer + 1
     * current layer  = j = layer
     * previous layer = i = layer - 1
     */
    /* Start sigma calculation at output layer */
    for (int layer = ncConfig->hidden_layers; layer >= 0; layer--) {
        for (int neuronC = 0; neuronC < ncConfig->arch.topology[layer + 1].neurons; neuronC++) {
            /*Output layer uses the network output to determine the error signal,
            therefore the program branches here
                */
            if (layer == ncConfig->hidden_layers) {
                float sigma = (ncConfig->setpoint - pInput[0]) * ncConfig->arch.topology[layer].activation->backward(neuron[layer][neuronC].netinput);
                neuron[layer][neuronC].sigma = sigma;
                n++;
                /* Hidden layers use previous layer's error signal to determine their error signal */
            } else {
                float errorSum = 0;
                for (int k = 0; k < ncConfig->arch.topology[layer + 2].neurons; k++) {
                    errorSum += neuron[layer + 1][k].sigma * weight[layer + 1][neuronC][k];
                }
                float sigma = errorSum * ncConfig->arch.topology[layer].activation->backward(neuron[layer][neuronC].netinput);
                neuron[layer][neuronC].sigma = sigma;
                n++;
            }
        }
    }
    /* Sanity check to see if all sigma values of all neurons have been calculated */
    assert(n == ncConfig->arch.total_neurons);
}

static void updateWeightsAndBiases(neuralControllerConfig_st* ncConfig, control_st *control, float ***weight, neuron_st **neuron){
    int n = 0; /* Sanity check variable to count neurons during update step */
    int w = 0; /* Sanity check variable to count weights during update step */

    /*Backpropagation*/
    /*For detailed expflaination see https://en.wikipedia.org/wiki/Backpropagation
     * next layer     = k = layer + 1
     * current layer  = j = layer
     * previous layer = i = layer - 1
     */
    for (int layer = ncConfig->hidden_layers; layer >= 0; layer--) {
        for (int neuronC = 0; neuronC < ncConfig->arch.topology[layer + 1].neurons; neuronC++) {
            /*Output layer uses the rating to determine the error signal,
            therefore the program branches here
                */
            if (layer == ncConfig->hidden_layers) {
                for (int k = 0; k < ncConfig->arch.topology[layer].neurons; k++) {
                    weight[layer][k][neuronC] += ncConfig->learning_rate * neuron[layer][neuronC].sigma * neuron[layer - 1][k].netoutput;
                    w++;
                }
                neuron[layer][neuronC].bias += ncConfig->learning_rate * neuron[layer][neuronC].sigma;
                n++;
            } else {
                for (int k = 0; k < ncConfig->arch.topology[layer].neurons; k++) {
                    if (layer > 0)
                        weight[layer][k][neuronC] += ncConfig->learning_rate * neuron[layer][neuronC].sigma * neuron[layer - 1][k].netoutput;
                    else
                        weight[layer][k][neuronC] += ncConfig->learning_rate * neuron[layer][neuronC].sigma * control->input[k];
                    w++;
                }
                neuron[layer][neuronC].bias += ncConfig->learning_rate * neuron[layer][neuronC].sigma;
                n++;
            }
        }
    }
    assert(w == ncConfig->arch.total_weights);
    assert(n == ncConfig->arch.total_neurons);
}

float stepDecayLr(int epoch, float initialLr, int stepSize, float gamma) {
    int decay_steps = epoch / stepSize;
    float learningRateRel = initialLr * powf(gamma, (float)decay_steps);
    return clampf(learningRateRel, initialLr*gamma, initialLr);
}

void saveArrayToFile(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening model file");
        exit(EXIT_FAILURE);
    }

    fclose(file);
}

static const activation_st *get_activation(activation_type_en id) {
    switch (id) {
        case ACTIVATION_SIGMOID:    return &ACT_SIGMOID;
        case ACTIVATION_TANH:       return &ACT_tanhf;
        case ACTIVATION_LEAKY_RELU: return &ACT_LEAKY_RELU;
        case ACTIVATION_LINEAR:     return &ACT_LINEAR;
        default:                    return &ACT_LINEAR; // or handle error
    }
}

static inline float clampf(float value, float min, float max)
{
    if (value < min) {
        return min;
    }

    if (value > max) {
        return max;
    }

    return value;
}