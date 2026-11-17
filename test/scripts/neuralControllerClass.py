# SPDX-FileCopyrightText: 2023-2026 Jakob Schatzl
# SPDX-License-Identifier: Apache-2.0

# @file neuralControllerClass.py
# @author Jakob Schatzl
# @brief Python interface for the neural controller
# @version 1.0
# @date 2026-11-17

import ctypes
import random
import math
import sys
from enum import IntEnum
from pathlib import Path

_LIB_EXT = ".dll" if sys.platform == "win32" else ".so"
_LIB_PATH = Path(__file__).resolve().parents[2] / "build" / "shared" / f"neuralControllerInterface{_LIB_EXT}"
lib = ctypes.CDLL(str(_LIB_PATH))

activation_st = ctypes.c_void_p

class activation_type_en(ctypes.c_int):
    ACTIVATION_SIGMOID    = 0
    ACTIVATION_TANH       = 1
    ACTIVATION_LEAKY_RELU = 2
    ACTIVATION_LINEAR     = 3

class layer_st(ctypes.Structure):
    _fields_ = [
        ("neurons", ctypes.c_int),
        ("activation", activation_st),  # const activation_st*
    ]

class arch_st(ctypes.Structure):
    _fields_ = [
        ("topology", ctypes.POINTER(layer_st)),
        ("total_neurons", ctypes.c_int),
        ("total_weights", ctypes.c_int),
    ]

class neuralControllerConfig_st(ctypes.Structure):
    _fields_ = [
        ("hidden_layers",        ctypes.c_int),
        ("layers",               ctypes.c_int),
        ("neurons",              ctypes.c_int),
        ("output_layer_neurons", ctypes.c_int),
        ("inputs",               ctypes.c_int),
        ("initialized",          ctypes.c_int),
        ("learning_rate",        ctypes.c_float),
        ("setpoint",             ctypes.c_float),
        ("isJordan",             ctypes.c_bool),
        ("activation_hidden",    activation_type_en),
        ("activation_output",    activation_type_en),
        ("arch",                 arch_st),
    ]

class control_st(ctypes.Structure):
    _fields_ = [
        ("act_old", ctypes.c_float),
        ("act_new", ctypes.c_float),
        ("rating", ctypes.c_float),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("input_old", ctypes.POINTER(ctypes.c_float)),
        ("epoch", ctypes.c_uint32),
    ]

class neuron_st(ctypes.Structure):
    _fields_ = [
        ("netinput", ctypes.c_float),
        ("netoutput", ctypes.c_float),
        ("bias", ctypes.c_float),
        ("sigma", ctypes.c_float),
    ]

# FCTPTR_TYPE = ctypes.CFUNCTYPE(ctypes.c_float)

FCTPTR_TYPE = ctypes.CFUNCTYPE(
    ctypes.c_float,  # return type: float
    ctypes.c_float,  # first argument: float
    ctypes.c_float,  # second argument: float
)

@FCTPTR_TYPE
def random_ctypes_float(fanIn: float, fanOut: float) -> float:
    return random.uniform(a=0.01, b=0.1)

@FCTPTR_TYPE
def xavier_ctypes_float(fanIn: float, fanOut: float) -> float:
    scale = 0.2
    limit = scale * (6.0 / (fanIn + fanOut)) ** 0.5
    return random.uniform(-limit, limit)

@FCTPTR_TYPE
def gauss_ctypes_float(fanIn: float, fanOut: float) -> float:
    scale = 0.1
    stddev = scale / math.sqrt(fanIn)
    return random.gauss(mu=0.0, sigma=0.1)

class InitMethod(IntEnum):
    RANDOM = 0
    GAUSS = 1
    XAVIER = 2

INIT_METHODS = {
    InitMethod.RANDOM: random_ctypes_float,
    InitMethod.GAUSS: gauss_ctypes_float,
    InitMethod.XAVIER: xavier_ctypes_float,
}

class NeuralController:
    lib.neuralController_Init.restype  = ctypes.c_int
    lib.neuralController_Init.argtypes = [
        ctypes.POINTER(neuralControllerConfig_st),                                          # ncConfig
        ctypes.POINTER(control_st),                                                         # control
        FCTPTR_TYPE,                                                                        # fctPtr
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))),    # pWeight
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(neuron_st))),                          # pNeuron
    ]

    lib.neuralController_Run.restype = ctypes.c_int
    lib.neuralController_Run.argtypes = [
        ctypes.POINTER(neuralControllerConfig_st),                                      # ncConfig
        ctypes.POINTER(control_st),                                                     # control
        ctypes.POINTER(ctypes.c_float),                                                # pOutput
        ctypes.POINTER(ctypes.c_float),                                                # pInput
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),                # weight
        ctypes.POINTER(ctypes.POINTER(neuron_st)),                                      # neuron
    ]

    lib.neuralController_Free.argtypes = [
        ctypes.POINTER(neuralControllerConfig_st),                                      # ncConfig
        ctypes.POINTER(control_st),                                                     # control
        ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),                # weight
        ctypes.POINTER(ctypes.POINTER(neuron_st)),                                      # neuron
    ]

    def __init__(self, hidden_layers: int, neurons: int, setpoint: float, isJordan: bool, init_method: int):
        self.ncConfig = neuralControllerConfig_st(
            hidden_layers = hidden_layers,
            layers = hidden_layers+2,
            neurons = neurons,
            output_layer_neurons = 1,
            inputs = 2,
            initialized = 0,
            activation_hidden = activation_type_en.ACTIVATION_TANH,
            activation_output = activation_type_en.ACTIVATION_LINEAR,
            setpoint = setpoint,
            isJordan = isJordan,
            arch = arch_st(),
        )
        self.control = control_st()
        self.control.epoch = 0

        self.weightPtr = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        self.weights = self.weightPtr()

        self.neuronPtr = ctypes.POINTER(ctypes.POINTER(neuron_st))
        self.neurons = self.neuronPtr()

        self.result = lib.neuralController_Init(
        ctypes.byref(self.ncConfig),
        ctypes.byref(self.control),
        INIT_METHODS[init_method],
        ctypes.byref(self.weights),
        ctypes.byref(self.neurons),
        )
        self.yn = (ctypes.c_float)()
        self.output = (ctypes.c_float)()

    def run(self, input: float, learning_rate: float, setpoint: float) -> float:
        # self.control.input[0] = ctypes.c_float(input)
        nnInput = ctypes.c_float(float(input))
        self.ncConfig.learning_rate = learning_rate
        self.ncConfig.setpoint = setpoint
        lib.neuralController_Run(ctypes.byref(self.ncConfig), ctypes.byref(self.control), ctypes.byref(self.output), ctypes.byref(nnInput), self.weights, self.neurons)
        return self.output.value

    def __del__(self):
        lib.neuralController_Free(ctypes.byref(self.ncConfig), ctypes.byref(self.control), self.weights, self.neurons)
