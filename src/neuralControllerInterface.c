#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <neuralController.h>

typedef neuralControllerConfig_st NeuralControllerConfigObject;

static PyObject *NeuralController_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)args;
    (void)kwds;
    NeuralControllerConfigObject *self = (NeuralControllerConfigObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->hidden_layers = 0;
        self->layers = 0;
        self->neurons = 0;
        self->output_layer_neurons = 0;
        self->inputs = 0;
        self->max_epochs = 0;
        self->learning_rate = 0.0;
        self->setpoint = 0.0;
        self->arch.topology = NULL;
        self->arch.total_neurons = 0;
        self->arch.total_weights = 0;
    }
    return (PyObject *)self;
}

static int NeuralController_init(NeuralControllerConfigObject *self, PyObject *args, PyObject *kwds){
    static char *kwlist[] = {
        "hidden_layers", "layers", "neurons", "output_layer_neurons", "inputs", "max_epochs", 
        "learning_rate", "setpoint", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiiiiidd", kwlist,
                                     &self->hidden_layers, &self->layers, &self->neurons, &self->output_layer_neurons,
                                     &self->inputs, &self->max_epochs, &self->learning_rate, &self->setpoint)){
        return -1;
                                     }

    self->arch.topology = NULL;
    self->arch.total_neurons = 0;
    self->arch.total_weights = 0;

    return 0;
}

static PyMethodDef NeuralControllerTypeMethods[] = {
    {NULL, NULL, 0, NULL}
};

static PyMethodDef NeuralControllerModuleMethods[] = {
    {NULL, NULL, 0, NULL}
};

static PyTypeObject NeuralControllerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "neuralController.NeuralController",
    .tp_doc = "Neural Controller Python module",
    .tp_basicsize = sizeof(NeuralControllerConfigObject),
    .tp_itemsize = 0,
    .tp_methods = NeuralControllerTypeMethods,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NeuralController_new,
    .tp_init = (initproc)NeuralController_init,
};

static struct PyModuleDef neuralControllerInterfaceModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "neuralControllerInterface",
    .m_doc = "Neural controller Python interface",
    .m_methods = NeuralControllerModuleMethods,
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_neuralControllerInterface(void) {
    PyObject *module;
    if (PyType_Ready(&NeuralControllerType) < 0){
        return NULL;
    }
    module = PyModule_Create(&neuralControllerInterfaceModule);
    if(module == NULL){
        return NULL;
    }
    Py_INCREF(&NeuralControllerType);
    if(PyModule_AddObjectRef(module, "NeuralController", (PyObject *)&NeuralControllerType) < 0){
        Py_XDECREF(&NeuralControllerType);
        Py_XDECREF(module);
        return NULL;
    }
    return module;
}
