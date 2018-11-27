linreg: src/linreg.cu
	mkdir obj
	nvcc -o obj/linreg src/linreg.cu
	nvcc -Xcompiler -fPIC -shared -o obj/linreg.so src/linreg.cu

make clean:
	rm obj/linreg.so
	rm obj/linreg
	rm obj
