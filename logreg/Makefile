log_reg: log_reg.cu vec_kernels.cu
	nvcc -O3 -o log_reg log_reg.cu vec_kernels.cu
	nvcc -O3 -Xcompiler -fPIC -shared -o log_reg.so log_reg.cu vec_kernels.cu

clean:
	rm -rf __pycache__
	rm -rf env
	rm -rf log_reg
	rm -rf log_reg.so
