main: src/main.cpp Graph.o Timer.o
	mkdir -p bin
	g++ -O2 -o bin/main src/main.cpp Graph.o Timer.o -fopenmp

Graph.o: src/Graph.cpp
	g++ -O2 -c -o Graph.o src/Graph.cpp -fopenmp

Timer.o: src/Timer.cpp
	g++ -O2 -c -o Timer.o src/Timer.cpp

clean:
	'rm' *.o