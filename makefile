myPaint: main.o pgmProcess.o pgmUtility.o
	nvcc -arch=sm_86 -o hw2 main.o pgmProcess.o pgmUtility.o -I.

main.o: main.cu
	nvcc -arch=sm_86 -c main.cu

pgmProcess.o: pgmProcess.cu pgmProcess.h
	nvcc -arch=sm_86 -c pgmProcess.cu

pgmUtility.o: pgmUtility.cu pgmUtility.h
	nvcc -arch=sm_86 -c pgmUtility.cu

#seqPgmUtility.o: seqPgmUtility.c seqPgmUtility.h
#	gcc -c -o seqPgmUtility.o seqPgmUtility.c

clean:
	rm -r *.o hw2
