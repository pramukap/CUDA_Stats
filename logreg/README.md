# Logistic Regression
This subdirectory contains a CUDA-based implementation of the Logistic Regression algorithm, with a Python-based implementation for comparison. 
## Setup
This project assumes that you have `python3`, `nvcc`, and `virtualenv` installed on your machine.

```
nvcc -Xcompiler -fPIC -shared -o log_reg.so log_reg.cu matrixFunctions.cu vec_kernels.cu

virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Running
All testing for this project was done within a Jupyter notebook, which is a Python execution visualization available through the web browser. To launch, execute `jupyter notebook` from the `env` that was created in the setup section. After that, open the Jupyter notebook file listed here and run the cells to test.