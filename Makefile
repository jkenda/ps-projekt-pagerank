main: src/main.cpp Matrix.o
	g++ -g -o dist/main src/main.cpp Matrix.o -fopenmp

Matrix.o: src/Matrix.cpp
	g++ -g -c -o Matrix.o src/Matrix.cpp -fopenmp

clean:
	'rm' *.o