#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

#define ERROR_MESSAGE "Usage:\n-e edgeWidth  oldImageFile  newImageFile\n-c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n-l p1row p1col p2row p2col oldImageFile newImageFile\n"
FILE* fileOpen(const char *filename, const char *mode);
FILE* fileClose(FILE *fp);

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s -c|-e|-l [additional parameters]\n", argv[0]);
        return -1;
    }

    char *header[rowsInHeader];
    const char *outputHeader[rowsInHeader];
    int numRows, numCols;
    char oldImageFile[256] = "";
    char newImageFile[256] = "";
    FILE *fileInput;
    FILE *fileOutput;
    int *pixels;

    switch (argv[1][1]) {
        case 'c': {

            if (argc != 7) {
                printf("%s", ERROR_MESSAGE);
                return -1;
            }

            int circleCenterRow = atoi(argv[2]);
            int circleCenterCol = atoi(argv[3]);
            int radius = atoi(argv[4]);
            strcpy(oldImageFile, argv[5]);
            strcpy(newImageFile, argv[6]);

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }
            
            pgmDrawCircle(pixels, numRows, numCols, circleCenterRow, circleCenterCol, radius, header);

            pgmWrite(outputHeader, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);

            break;
        }
        case 'e': {

            if (argc != 5) {
                printf("%s", ERROR_MESSAGE);
                return -1;
            }

            int edgeWidth = atoi(argv[2]);
            strcpy(oldImageFile, argv[3]);
            strcpy(newImageFile, argv[4]);

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }

            pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);

            pgmWrite(outputHeader, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);
            break;
        }
        case 'l': {

            if (argc != 8) {
                printf("%s", ERROR_MESSAGE);
                return -1;
            }

            int p1row = atoi(argv[2]);
            int p1col = atoi(argv[3]);
            int p2row = atoi(argv[4]);
            int p2col = atoi(argv[5]);
            strcpy(oldImageFile, argv[6]);
            strcpy(newImageFile, argv[7]);

            fileInput = fileOpen(oldImageFile, "r");
            fileOutput = fileOpen(newImageFile, "w");

            pixels = pgmRead(header, &numRows, &numCols, fileInput);

            if (pixels == NULL) {
                printf("Error reading PGM file: %s\n", oldImageFile);
                fileClose(fileInput);
                fileClose(fileOutput);
                return -1;
            }

            pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);

            pgmWrite(outputHeader, pixels, numRows, numCols, fileOutput);

            fileClose(fileInput);
            fileClose(fileOutput);
            break;

        }
        default: {
            printf(ERROR_MESSAGE);
            return -1;
        }
    }
    return 0;
}

FILE* fileOpen(const char *filename, const char *mode) {
    FILE *fp = fopen(filename, mode);
    if (fp == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return fp;
}

FILE* fileClose(FILE *fp) {
    if (fclose(fp) != 0) {
        printf("Error closing file\n");
        exit(EXIT_FAILURE);
    }
    return NULL;
}