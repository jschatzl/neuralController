# Neuro Controller
Implementation of a feed forward neural network with online learning through backpropagation and online learning that controls a simulated PTn plant. 

## Background

This project originated as a proof-of-concept developed as part of a Master's thesis.

The source code was developed by Jakob Schatzl and is provided as a standalone implementation for further development and reuse.

## Folder structure
```text
├── build                       CMake's build folder, configured through CMakePreset.json
├── doc                         Several .tex and .puml files that were used in the corresponding Master Thesis        
├── inc                         Include directory
├── scripts                     Script folder for the Python testing
├── src                         Source folder
│   ├── activation.c            Implementation of the activation functions
│   ├── main.c                  Very rough self-test routine for the neural controller
│   ├── neuralController.c      The main algorithm, also documented with public API
│   ├── pid.c                   A implementation for a PID controller
│   ├── pt2.c                   A C-simulation for a PT2 plant
├── CMakeLists.txt              Declaration of targets
├── CMakePreset.json            Declaration of CMake presets
├── README.md                   This file
```

## Neural Network
The network code is found in src/neuralController.c and the corresponding header file neuralController.h
Using CMake either a full executable using the src/main.c can be built, a shared library that is then consumed by the tests scripts in scripts/ or a STATIC CMake
target that can be consumed by another exectuable.
Consume with 
```
target_link_libraries(myTarget
    PUBLIC
        neuralController
)
```

## Building
There are 2 presets that can be used to build the target, so either use the normal preset for the Python testing that creates a shared library
(build/shared/neuralControllerInterface.so on Linux, .dll on Windows) that is loaded by the Python scripts. The VS Code launch configuration
"Python: Test Neural Controller" builds it automatically if it does not exist yet:
```
cmake --preset shared
cmake --build build/shared
```
or use the self-test preset which compiles the main.c for a quick test to see if everything works or execute together with valgrind for memory leak verification
```
cmake --preset test
cmake --build build/test
```

## Dependencies

The core implementation is written in C and uses the standard C library.

Python scripts used for testing and evaluation require:

- NumPy
- Matplotlib

These dependencies are not included in this repository.

## License
Copyright © 2023-2026 Jakob Schatzl.

This project is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for the full license text.