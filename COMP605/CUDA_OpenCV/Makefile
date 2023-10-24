hw4:
	nvcc rgb2grey.cu -I/home/perimbeti/build/opencv/include/opencv4 -L/home/perimbeti/build/opencv/lib -lstdc++ -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -w -o hw4

link:
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/perimbeti/build/opencv/lib

clean:
	rm hw4
	rm greyscale_bear.jpg
