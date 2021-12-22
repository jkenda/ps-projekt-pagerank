CPPC= g++
FLAGS= -O2 -fopenmp -s -mtune=native


all: main.o Graph.o Timer.o
	mkdir -p bin
	$(CPPC) $(FLAGS) -o bin/main main.o Graph.o Timer.o

main.o: src/main.cpp
	$(CPPC) $(FLAGS) -c -o main.o src/main.cpp

Graph.o: src/Graph.cpp
	$(CPPC) $(FLAGS) -c -o Graph.o src/Graph.cpp

Timer.o: src/Timer.cpp
	$(CPPC) $(FLAGS) -c -o Timer.o src/Timer.cpp

clean:
	'rm' *.o