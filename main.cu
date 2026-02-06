#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include "pgmUtility.h"

extern "C" {
#include "timing.h"
#include "seqPgmUtility.h"
}

#define ERROR_MESSAGE "Usage:\n-e edgeWidth  oldImageFile  newImageFile\n-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n-l p1row p1col p2row p2col oldImageFile newImageFile\n"
FILE* fileOpen(const char *filename, const char *mode);
FILE* fileClose(FILE *fp);

int main(int argc, char *argv[]) {
    int status = 0;

    if (argc < 2) {
        printf("%s", ERROR_MESSAGE);
        return -1;
    }

    char *header[rowsInHeader];
    for (int i = 0; i < rowsInHeader; i++) {
        header[i] = (char*)malloc(maxSizeHeadRow);
        if (!header[i]) { perror("malloc"); exit(1); }
    }
    int numRows, numCols;
    char oldImageFile[256] = "";
    char newImageFile[256] = "";
    FILE *fileInput = NULL;
    FILE *fileOutput = NULL;

    int *pixels = NULL;
    int **sequentialPixels = NULL;
    double now, then;
    double scost, pcost;

    File *fptr = Null;
    fptr = fopen("Times.txt", "w");


    switch (argv[1][1]) {
        case 'c': {

            if (argc != 7) {
                printf("%s", ERROR_MESSAGE);
                status = -1;
                break;
            }

            int circleCenterRow = atoi(argv[2]);
            int circleCenterCol = atoi(argv[3]);
            int radius = atoi(argv[4]);
            strcpy(oldImageFile, argv[5]);
            strcpy(newImageFile, argv[6]);

            //Time Serial Solution
            then = currentTime();
            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            sequentialPixels = seqPgmRead(header, &numRows, &numCols, fileInput);
            if (sequentialPixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }

            seqPgmDrawCircle(sequentialPixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);
            seqPgmWrite(header, sequentialPixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            scost = now - then;

            //printf("Serial code execution time for circle drawing in second is %lf\n", scost);



            //Time Parallel Solution
            then = currentTime();

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }
            
            pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);

            pgmWrite((const char**)header, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            pcost = now - then;
            
            if (fptr == NULL) {
                printf("The file is not opened.");
            } else {
                printf("The file is now opened.\n");
                
                fprintf(fptr,"Serial code execution time for circle drawing in second is %lf\n", scost);
                fprintf(fptr,"%%%%%% Parallel code execution time for circle drawing is %lf\n", pcost);
                fprintf(fptr,"%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
                fprintf(fptr,"%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            }

            /*
            printf("%%%%%% Parallel code execution time for circle drawing is %lf\n", pcost);

            printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
            printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            */

            break;
        }
        case 'e': {

            if (argc != 5) {
                printf("%s", ERROR_MESSAGE);
                status = -1;
                break;
            }

            int edgeWidth = atoi(argv[2]);
            strcpy(oldImageFile, argv[3]);
            strcpy(newImageFile, argv[4]);

            //Time Serial Solution
            then = currentTime();
            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            sequentialPixels = seqPgmRead(header, &numRows, &numCols, fileInput);
            if (sequentialPixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }

            seqPgmDrawEdge(sequentialPixels, numRows, numCols, edgeWidth, header);
            seqPgmWrite(header, sequentialPixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            scost = now - then;

            //printf("Serial code execution time for edge drawing in second is %lf\n", scost);



            //Time Parallel Solution
            then = currentTime();

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }

            pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);

            pgmWrite((const char**)header, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            pcost = now - then;

            if (fptr == NULL) {
                printf("The file is not opened.");
            } else {
                printf("The file is now opened.\n");
                
                fprintf(fptr,"Serial code execution time for edge drawing in second is %lf\n", scost);
                fprintf(fptr,"%%%%%% Parallel code execution time for edge drawing is %lf\n", pcost);
                fprintf(fptr,"%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
                fprintf(fptr,"%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            }

            /*
            printf("%%%%%% Parallel code execution time for edge drawing is %lf\n", pcost);
            printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
            printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            */
            break;
        }
        case 'l': {

            if (argc != 8) {
                printf("%s", ERROR_MESSAGE);
                status = -1;
                break;
            }

            int p1row = atoi(argv[2]);
            int p1col = atoi(argv[3]);
            int p2row = atoi(argv[4]);
            int p2col = atoi(argv[5]);
            strcpy(oldImageFile, argv[6]);
            strcpy(newImageFile, argv[7]);

            //Time Serial Solution
            then = currentTime();
            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            sequentialPixels = seqPgmRead(header, &numRows, &numCols, fileInput);
            if (sequentialPixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }

            seqPgmDrawLine(sequentialPixels, numRows, numCols, header, p1row, p1col, p2row, p2col);
            seqPgmWrite(header, sequentialPixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            scost = now - then;

            //printf("Serial code execution time for line drawing in second is %lf\n", scost);



            //Time Parallel Solution
            then = currentTime();

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                status = -1;
                break;
            }

            pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);

            pgmWrite((const char**)header, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            //End Time
            now = currentTime();
            pcost = now - then;

            if (fptr == NULL) {
                printf("The file is not opened.");
            } else {
                printf("The file is now opened.\n");
                
                fprintf(fptr,"Serial code execution time for line drawing in second is %lf\n", scost);
                fprintf(fptr,"%%%%%% Parallel code execution time for line drawing is %lf\n", pcost);
                fprintf(fptr,"%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
                fprintf(fptr,"%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            }
            
            /*
            printf("%%%%%% Parallel code execution time for line drawing is %lf\n", pcost);

            printf("%%%%%% The speedup(SerialTimeCost / ParallelTimeCost) when using GPU is %lf\n", scost / pcost);
            printf("%%%%%% The efficiency(Speedup / NumProcessorCores) when using GPU is %lf\n", scost / pcost / 4);
            */

            break;

        }
        default: {
            printf(ERROR_MESSAGE);
            status = -1;
            break;
        }
    }

    // Free pixels if allocated
    free(pixels);

    // Free header buffers
    for (int i = 0; i < rowsInHeader; i++) {
        free(header[i]);
    }

    return status;
}

FILE* fileOpen(const char *filename, const char *mode) {
    FILE *fp = fopen(filename, mode);
    if (fp == NULL) {
        printf("%s", ERROR_MESSAGE);
        exit(EXIT_FAILURE);
    }
    return fp;
}

FILE* fileClose(FILE *fp) {
    if (fclose(fp) != 0) {
        printf("%s", ERROR_MESSAGE);
        exit(EXIT_FAILURE);
    }
    return NULL;
}