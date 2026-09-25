/**
 * @file neural_controller.c
 * @author Jakob Schatzl
 * @brief Implementation of library functions for the neural controller
 * @version 0.1
 * @date 2023-01-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "neuralController.h"

int neuralController_Init(neuralControllerConfig_st* ncConfig, control_st *control, float (*fctPtr)(), double**** pWeight, neuron_st*** pNeuron) {
    control->act_new = 0;
    control->act_old = ncConfig->setpoint - 0;
    control->input = (double*)calloc(ncConfig->inputs, sizeof(double));
    control->input_old = (double*)calloc(ncConfig->inputs, sizeof(double));
    control->rating = 0;

    if(ncConfig->isJordan){
        ncConfig->inputs += ncConfig->output_layer_neurons;
    }

    // double *error_array = calloc(ncConfig->max_epochs, sizeof(double));
    ncConfig->arch.total_neurons = ncConfig->neurons * ncConfig->hidden_layers + ncConfig->output_layer_neurons;
    ncConfig->arch.total_weights = (ncConfig->inputs * ncConfig->neurons) + (ncConfig->neurons * ncConfig->neurons * (ncConfig->hidden_layers - 1)) + (ncConfig->neurons * ncConfig->output_layer_neurons);

    ncConfig->arch.topology = (int *)calloc(ncConfig->layers, sizeof(int));
    for (int i = 0; i < ncConfig->layers; i++) {
        if (i == ncConfig->layers - 1) {
            ncConfig->arch.topology[i] = ncConfig->output_layer_neurons;
        } else if (i == 0) {
            ncConfig->arch.topology[i] = ncConfig->inputs;
        } else {
            ncConfig->arch.topology[i] = ncConfig->neurons;
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
        fread(weight, sizeof(double), total_weight, file);
        fclose(file);
    }

    /*Initialize weight and bias with values from the .bin and
      initialize the rest with 0*/
    for (int layer = 0; layer < ncConfig->layers; layer++) {
        for (int j = 0; j < ncConfig->arch.topology[layer]; j++) {
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer][j].bias = (double)(*fctPtr)();
            neuron[layer][j].netinput = 0.0;
            neuron[layer][j].netoutput = 0.0;
            neuron[layer][j].sigma = 0.0;
        }
    }
    ncConfig->initialized = 1;

#else /*LOAD_weight*/

    /*Initialize weight and bias with random values between 0 and 1 and
      initialize the rest with 0*/
    double ***weight = (double ***)calloc(ncConfig->layers - 1, sizeof(double **));
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        weight[layer] = (double **)calloc(ncConfig->arch.topology[layer], sizeof(double *));
        for (int j = 0; j < ncConfig->arch.topology[layer]; j++) {
            weight[layer][j] = (double *)calloc(ncConfig->arch.topology[layer + 1], sizeof(double));
            for (int k = 0; k < ncConfig->arch.topology[layer + 1]; k++) {
                weight[layer][j][k] = (double)(*fctPtr)();
            }
        }
    }

    neuron_st **neuron = (neuron_st **)calloc(ncConfig->layers, sizeof(neuron_st *));
    for (int layer = 1; layer < ncConfig->layers; layer++) {
        neuron[layer - 1] = (neuron_st *)calloc(ncConfig->arch.topology[layer], sizeof(neuron_st));
        for (int j = 0; j < ncConfig->arch.topology[layer]; j++) {
            /*Initialize bias and everything else in the neuron struct*/
            neuron[layer - 1][j].bias = (double)(*fctPtr)();
            neuron[layer - 1][j].netinput = 0.0;
            neuron[layer - 1][j].netoutput = 0.0;
            neuron[layer - 1][j].sigma = 0.0;
        }
    }
    *pWeight = weight;
    *pNeuron = neuron;
    ncConfig->initialized = 1;
    
#endif /*LOAD_weight*/

    return 42;
}

int neuralController_Run(neuralControllerConfig_st* ncConfig, control_st *control, double* pOutput, double* pInput, double*** weight, neuron_st** neuron) {
    int n = 0;
    int w = 0;
    float d2 = 0;

    control->input[0] = ncConfig->setpoint - pInput[0];
    for (int input_cnt = 0; input_cnt < ncConfig->inputs - 1; input_cnt++) {
        control->input[input_cnt + 1] = pInput[input_cnt];
    }

    /* Loop back output as input for Jordan network type */
    if(ncConfig->isJordan){
        control->input[ncConfig->inputs-1] = *pOutput;
    }
    /*Forward pass*/
    for (int layer = 0; layer < ncConfig->layers - 1; layer++) {
        for (int j = 0; j < ncConfig->arch.topology[layer + 1]; j++) {
            /*First hidden layer*/
            // double sum = neuron[layer][j].bias;
            double sum = 0;
            for (int k = 0; k < ncConfig->arch.topology[layer]; k++) {
                if (layer == 0)
                    sum += control->input[k] * weight[layer][k][j];
                else
                    sum += neuron[layer - 1][k].netoutput * weight[layer][k][j];
                //printf("Sum: %f W: %f ", sum, weight[layer][k][j]);
            }
            neuron[layer][j].netinput = sum;
            if (layer == ncConfig->hidden_layers)
                neuron[layer][j].netoutput = tanh(sum);
            else
                neuron[layer][j].netoutput = tanh(sum);
            n++;
            //printf("Sum: %f N: %f \n", neuron[layer][j].netinput, neuron[layer][j].netoutput);
        }
    }
    assert(n == ncConfig->arch.total_neurons);
    n = 0;

    // d2 = control->input[ncConfig->inputs - 1] - control->input_old[ncConfig->inputs - 1];
    // memcpy(control->input_old, &control->input, ncConfig->inputs);

    // control->act_new = ncConfig->setpoint - control->input[1];
    // control->rating = (fabs(control->act_new) - fabs(control->act_old)) + control->act_new;
    // control->act_old = control->act_new;
    /*Backpropagation*/
    /*For detailed explaination see https://en.wikipedia.org/wiki/Backpropagation
    /**
     * next layer     = k = layer + 1
     * current layer  = j = layer
     * previous layer = i = layer - 1
     */
    /*Start at output layer*/
    for (int layer = ncConfig->hidden_layers; layer >= 0; layer--) {
        for (int neuronC = 0; neuronC < ncConfig->arch.topology[layer + 1]; neuronC++) {
            /*Output layer uses the rating to determine the error signal,
            therefore the program branches here
             */
            if (layer == ncConfig->hidden_layers) {
                double sigma = (ncConfig->setpoint - neuron[ncConfig->hidden_layers][0].netoutput) * dTanh(neuron[layer][neuronC].netinput);
                // double sigma = control->rating * dTanh(neuron[layer][neuronC].netinput);
                for (int k = 0; k < ncConfig->arch.topology[layer]; k++) {
                    weight[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    w++;
                }
                neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                neuron[layer][neuronC].sigma = sigma;
                n++;
            } else {
                double errorSum = 0;
                for (int k = 0; k < ncConfig->arch.topology[layer + 2]; k++) {
                    errorSum += neuron[layer + 1][k].sigma * weight[layer + 1][neuronC][k];
                }
                double sigma = errorSum * dTanh(neuron[layer][neuronC].netinput);
                for (int k = 0; k < ncConfig->arch.topology[layer]; k++) {
                    if (layer > 0)
                        weight[layer][k][neuronC] += ncConfig->learning_rate * sigma * neuron[layer - 1][k].netoutput;
                    else
                        weight[layer][k][neuronC] += ncConfig->learning_rate * sigma * control->input[k];
                    w++;
                }
                neuron[layer][neuronC].bias += ncConfig->learning_rate * sigma;
                neuron[layer][neuronC].sigma = sigma;
                n++;
            }
        }
    }
    assert(w == ncConfig->arch.total_weights);
    assert(n == ncConfig->arch.total_neurons);
    w = 0;
    n = 0;
    control->epoch++;

    if((control->epoch % 1000) == 0){
        printf("Setpoint: %f, Learning rate: %f, Error: %f \n", ncConfig->setpoint, ncConfig->learning_rate, control->act_new);
    }
    *pOutput = neuron[ncConfig->hidden_layers][0].netoutput;
    return 3;
}

void neuralController_Free(neuralControllerConfig_st* ncConfig, control_st *control, double ***weight, neuron_st **neuron) {
    if((!weight) || (!neuron) || (!ncConfig))
        return;

    for(int layer = 0; layer < ncConfig->layers - 1; layer++){
        for(int j = 0; j < ncConfig->arch.topology[layer]; j++){
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

void saveArrayToFile(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening model file");
        exit(EXIT_FAILURE);
    }

    // Write the entire 3D array to the file
    //fwrite(weight, sizeof(double), total_weight, file);

    fclose(file);
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double dSigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double dTanh(double x) {
    double th = tanh(x);
    return 1.0 - th * th;
}
