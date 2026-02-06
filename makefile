myPaint: main.o pgmProcess.o seqPgmUtility.o pgmUtility.o timing.o
	nvcc -arch=sm_86 -o hw2 main.o timing.o pgmProcess.o seqPgmUtility.o pgmUtility.o -I.

main.o: main.cu
	nvcc -arch=sm_86 -c main.cu

pgmProcess.o: pgmProcess.cu pgmProcess.h
	nvcc -arch=sm_86 -c pgmProcess.cu

pgmUtility.o: pgmUtility.cu pgmUtility.h
	nvcc -arch=sm_86 -c pgmUtility.cu

timing.o: timing.c timing.h
	gcc -c -x c timing.c -l.

seqPgmUtility.o: seqPgmUtility.c seqPgmUtility.h
	gcc -c -x c seqPgmUtility.c -l.

clean:
	rm -r *.o hw2
