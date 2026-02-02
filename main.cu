#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s -c|-e|-l [additional parameters]\n", argv[0]);
        return -1;
    }

    char *header[rowsInHeader];
    int numRows, numCols;


    char *flag = argv[1];
    switch (flag[1]) {  // flag[1] is the character after the dash
        case 'c':

            if (argc != 7) {
                printf("Usage for circle: %s -c centerRow centerCol radius inputFile\n", argv[0]);
                return -1;
            }

            int circleCenterRow = atoi(argv[2]);
            int circleCenterCol = atoi(argv[3]);
            int radius = atoi(argv[4]);
            char oldImageFile[256] = argv[5];
            char newImageFile[256] = argv[6];

            fileptr *fileInput = fileOpen(oldImageFile, "r");
            fileptr *fileOutput = fileOpen(newImageFile, "w");

            int *pixels = pgmRead(argv[argc - 2], header, &numRows, &numCols);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }
            
            pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);

            pgmWrite(pixels, numRows, numCols, header, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            break;

        case 'e':

            if (argc != 5) {
                printf("Usage for edge: %s -e edgeWidth inputFile\n", argv[0]);
                return -1;
            }

            int edgeWidth = atoi(argv[2]);
            char oldImageFile[256] = argv[3];
            char newImageFile[256] = argv[4];

            fileptr *fileInput = fileOpen(oldImageFile, "r");
            fileptr *fileOutput = fileOpen(newImageFile, "w");

            int *pixels = pgmRead(argv[argc - 2], header, &numRows, &numCols);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }

            pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);

            pgmWrite(pixels, numRows, numCols, header, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);
            break;

        case 'l':

            if (argc != 8) {
                printf("Usage for line: %s -l p1row p1col p2row p2col inputFile\n", argv[0]);
                return -1;
            }

            int p1row = atoi(argv[2]);
            int p1col = atoi(argv[3]);
            int p2row = atoi(argv[4]);
            int p2col = atoi(argv[5]);
            char oldImageFile[256] = argv[6];
            char newImageFile[256] = argv[7];

            fileptr *fileInput = fileOpen(oldImageFile, "r");
            fileptr *fileOutput = fileOpen(newImageFile, "w");

            int *pixels = pgmRead(argv[argc - 2], header, &numRows, &numCols);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }

            pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);

            pgmWrite(pixels, numRows, numCols, header, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);
            break;

    }

    dim3 grid, block;
    block.x = 16;
    block.y = 16;
    grid.x = ceil((double)(numCols + block.x - 1) / block.x);
    grid.y = ceil((double)(numRows + block.y - 1) / block.y);

    drawCircleKernel<<<grid,block>>>(pixels, numRows, numCols, centerRow, centerCol, radius);


    drawEdgeKernel<<<grid,block>>>(pixels, numRows, numCols, edgeWidth);

    drawLineKernel<<<grid,block>>>(pixels, numRows, numCols, p1row, p1col, p2row, p2col);


    return 0;
}

*fileptr fileOpen(const char *filename, const char *mode) {
    fileptr fp = fopen(filename, mode);
    if (fp == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return fp;
}

*fileptr fileClose(fileptr fp) {
    if (fclose(fp) != 0) {
        printf("Error closing file\n");
        exit(EXIT_FAILURE);
    }
    return NULL;
}