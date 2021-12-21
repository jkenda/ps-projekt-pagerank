main: src/main.cpp Matrix.o
	mkdir -p bin
	g++ -O2 -o bin/main src/main.cpp Matrix.o -fopenmp

Matrix.o: src/Matrix.cpp
	g++ -O2 -c -o Matrix.o src/Matrix.cpp -fopenmp

clean:
	'rm' *.o