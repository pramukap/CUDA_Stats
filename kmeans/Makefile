kmeans: src/kmeans.cu src/kmeansHelper.cu
	mkdir obj
	nvcc -Xcompiler -fPIC -shared -o obj/kmeans.so src/kmeans.cu

make clean:
	rm obj/kmeans.so
	rmdir obj
