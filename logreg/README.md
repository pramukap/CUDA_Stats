# Logistic Regression
This subdirectory contains a CUDA-based implementation of the Logistic Regression algorithm, with a Python-based implementation for comparison. 
## Setup
This project assumes that you have `python3`, `nvcc`, and `virtualenv` installed on your machine.

```
make
./start_env.sh
```

## Running
Testing for this project was done within a Jupyter notebook, which is a Python execution visualization available through the web browser. To launch, execute `jupyter notebook` from the `env` that was created in the setup section. After that, open the Jupyter notebook file listed here and run the cells to test. The time analysis data comes from the C executable, which can be invoked by calling `./log_reg`.
