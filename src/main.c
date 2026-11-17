// SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.c
 * @author Jakob Schatzl
 * @brief Selft-test of the neural controller 
 * @version 1.0
 * @date 2026-11-17
 */
#include "main.h"
#include "activation.h"
#include "pt2.h"

pthread_t feedInputThread;
pthread_mutex_t mutex;
static PT2_Handle_t pt2;

/**
 * @brief Generates a random float between 0 and 0.1
 *
 *
 * @param
 * @return
 */
float generateRandomfloat(void);

float generateRandomfloatWithLimit(float limit);

float generateXavierfloat(float fanIn, float fanOut);

/**
 * @brief main loop for testing
 *
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, const char* argv[]) {
    // create the neural controller config
    neuralControllerConfig_st ncConfig;
    float yn = 0;                   // state of plant
    float output = 0.0;            // neural Network output
    float *input = NULL;

    // end pointer for strtol
    char* end;

    // initialize the neural controller config via command line arguments
    ncConfig.inputs = (int)strtol(argv[1], &end, 10);
    ncConfig.hidden_layers = (int)strtol(argv[2], &end, 10);
    ncConfig.layers = ncConfig.hidden_layers + 2;
    ncConfig.learning_rate = (float)roundf(strtof(argv[3], &end) * 100) / 100;
    unsigned int max_epochs = (int)strtol(argv[4], &end, 10);
    ncConfig.neurons = (int)strtol(argv[5], &end, 10);
    ncConfig.output_layer_neurons = (int)strtol(argv[6], &end, 10);
    ncConfig.setpoint = (float)roundf(strtof(argv[7], &end) * 100) / 100;
    ncConfig.activation_hidden = ACTIVATION_TANH;
    ncConfig.activation_output = ACTIVATION_TANH;
    long int temp = strtol(argv[8], &end, 10);
    ncConfig.isJordan = (_Bool)temp;

    printf("Jordan Network: %d\n", ncConfig.isJordan);

    input = (float*)calloc(ncConfig.inputs, sizeof(float));

    // Neural Network
    float ***weight = NULL;
    neuron_st **neuron = NULL; 
    control_st control = {0};

    // set seed for rand() function
    srand(time(NULL));

    // function pointer to a function that generates random floats
    float (*randFctPtr)(float, float);
    randFctPtr = &generateXavierfloat;

    // Initialize the neuralController
    neuralController_Init(&ncConfig, &control, randFctPtr, &weight, &neuron);

    PT2_Init(&pt2, 0.001f);

    // Loop for testing over n epochs
    for (int i = 0; i < max_epochs; i++) {
        // set input for the next run
        input[0] = yn;
        // Run through feed forward + backpropagation
        neuralController_Run(&ncConfig, &control, &output, input, weight, neuron);
        /*Calculate next state of the I plant*/
        // yn = i_plant(yn, output);
        yn = PT2_Update(&pt2, output);
        // debug print message
        if ((i % 100) == 0) {
            printf("Epoch: %d Plant output: %f Error: %f u: %f \n", i, yn, ncConfig.setpoint - yn, output);
        }
        if(i > 4000)
            ncConfig.learning_rate = 0.001;
        // if(i == (max_epochs/2)){
        //     ncConfig.setpoint = ncConfig.setpoint/2;
        // }
    }

#if PlotGraph
    // Create image reference structure
    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();

    // Plot
    DrawScatterPlot(imageRef, 600, 400, x_values, ncConfig.max_epochs, error_array, ncConfig.max_epochs, NULL);

    // Create PNG file and free allocations
    size_t length;
    float* pngData = ConvertToPNG(&length, imageRef->image);
    WriteToFile(pngData, length, "control.png");
    DeleteImage(imageRef->image);
    FreeAllocations();
#endif  // PlotGraph

#if WriteToFile

    FILE* file = fopen(filename, "r+");  // Open file in read+write mode
    if (file == NULL) {
        perror("Error opening results file");
        return 1;
    }else{
        int i = 0;
        char character;
        while ((character = fgetc(file)) != EOF) {
            if (character == '\n') {
                fprintf(file, ", %f\n", error_array[i]);
                i++;
            }
        }
    }

    fclose(file);

#endif  // WriteToFile

    //Write weights to file
    //saveArrayToFile("test.bin");
    free(input);

    neuralController_Free(&ncConfig, &control, weight, neuron);

    return 42;
}

/**
 * @brief Function that calculates the next step of an I plant
 *
 *
 * @param yn current state
 * @param u control variable
 * @return next state
 */
float i_plant(float yn, float u) {
    float K = 1;
    float T = 0.1;

    return yn + K * u * T;
}

float round_to_decimal_places(float value, int decimal_places) {
    float factor = powf(10, decimal_places);
    return round(value * factor) / factor;
}


float generateRandomfloat(void){
    return (rand() / (float)(RAND_MAX)) / 10;
}

float generateRandomfloatWithLimit(float limit) {
    float rnd = (float)rand() / (float)RAND_MAX;
    return (2.0f * rnd - 1.0f) * limit;
}

float generateXavierfloat(float fanIn, float fanOut){
    static float gain = 0.1;
    float limit = gain * sqrtf(6.0f / (fanIn + fanOut));
    return generateRandomfloatWithLimit(limit);
}