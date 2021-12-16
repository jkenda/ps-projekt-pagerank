main: main.cpp Matrix.o
	g++ -g -o main main.cpp Matrix.o -fopenmp

Matrix.o: Matrix.cpp
	g++ -g -c -o Matrix.o Matrix.cpp -fopenmp

clean:
	'rm' *.o