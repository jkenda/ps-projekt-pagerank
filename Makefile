CPPC= g++
FLAGS= -std=c++17 -O2 -fopenmp -s -mtune=native


all: main.o Graph.o Graph4CL.o Timer.o
	mkdir -p bin
	$(CPPC) $(FLAGS) -o bin/main main.o Graph.o Graph4CL.o Timer.o

main.o: src/main.cpp
	$(CPPC) $(FLAGS) -c -o main.o src/main.cpp

Graph.o: src/Graph.cpp src/Graph.hpp
	$(CPPC) $(FLAGS) -c -o Graph.o src/Graph.cpp

Graph4CL.o: src/Graph4CL.cpp src/Graph.hpp
	$(CPPC) $(FLAGS) -c -o Graph4CL.o src/Graph4CL.cpp

Timer.o: src/Timer.cpp
	$(CPPC) $(FLAGS) -c -o Timer.o src/Timer.cpp

clean:
	'rm' *.o

	
run:
	bin/main web-Google/web-Google.txt | tee output
	echo -n "\a"

run_slurm:	
	srun --reservation=fri bin/main web-Google/web-Google.txt | tee output
	echo -n "\a"

