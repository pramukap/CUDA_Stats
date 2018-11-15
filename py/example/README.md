# Example for calling C Function from Python

## Notes:
- Notice the `extern "C"` block encapsulating the function. This is required.
- Shows that `numpy` stores matrices in row-major order

## To compile
```
g++ -c -fpic print_matrix.cc
g++ -shared -o libprint.so print_matrix.o
```

## To run Python (assumes you have virtualenv)
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python print_matrix.py
```