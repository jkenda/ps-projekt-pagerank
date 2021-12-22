main: src/main.cpp Graph.o 
	mkdir -p bin
	g++ -O2 -o bin/main src/main.cpp Graph.o

Graph.o: src/Graph.cpp
	g++ -O2 -c -o Graph.o src/Graph.cpp

clean:
	'rm' *.o