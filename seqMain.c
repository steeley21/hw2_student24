#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "seqPgmUtility.h"

#define USAGE_MESSAGE \
"Usage:\n" \
" -e edgeWidth  oldImageFile  newImageFile\n" \
" -c circleCenterRow circleCenterCol radius  oldImageFile  newImageFile\n" \
" -l p1row p1col p2row p2col oldImageFile newImageFile\n"

static void printUsage(void) {
    fputs(USAGE_MESSAGE, stderr);
}

static int parseInt(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s == end || *end != '\0') return 0;           // not a pure int
    if (v < -2147483648L || v > 2147483647L) return 0; // int range
    *out = (int)v;
    return 1;
}

static void freeHeader(char **header) {
    for (int i = 0; i < rowsInHeader; i++) free(header[i]);
}

static void freePixels(int **pixels, int numRows) {
    if (!pixels) return;
    for (int r = 0; r < numRows; r++) free(pixels[r]);
    free(pixels);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printUsage();
        return 1;
    }

    const char *mode = argv[1];

    int centerRow, centerCol, radius;
    int edgeWidth;
    int p1row, p1col, p2row, p2col;
    const char *inName = NULL;
    const char *outName = NULL;

    enum { OP_NONE, OP_CIRCLE, OP_EDGE, OP_LINE } op = OP_NONE;

    if (strcmp(mode, "-c") == 0) {
        if (argc != 7) { printUsage(); return 1; }
        if (!parseInt(argv[2], &centerRow) ||
            !parseInt(argv[3], &centerCol) ||
            !parseInt(argv[4], &radius)) {
            printUsage(); return 1;
        }
        inName = argv[5];
        outName = argv[6];
        op = OP_CIRCLE;
    }
    else if (strcmp(mode, "-e") == 0) {
        if (argc != 5) { printUsage(); return 1; }
        if (!parseInt(argv[2], &edgeWidth)) { printUsage(); return 1; }
        inName = argv[3];
        outName = argv[4];
        op = OP_EDGE;
    }
    else if (strcmp(mode, "-l") == 0) {
        if (argc != 8) { printUsage(); return 1; }
        if (!parseInt(argv[2], &p1row) ||
            !parseInt(argv[3], &p1col) ||
            !parseInt(argv[4], &p2row) ||
            !parseInt(argv[5], &p2col)) {
            printUsage(); return 1;
        }
        inName = argv[6];
        outName = argv[7];
        op = OP_LINE;
    }
    else {
        printUsage();
        return 1;
    }

    FILE *in = fopen(inName, "r");
    if (!in) { perror("fopen input"); return 1; }

    FILE *out = fopen(outName, "w");
    if (!out) { perror("fopen output"); fclose(in); return 1; }

    // allocate header
    char *header[rowsInHeader];
    for (int i = 0; i < rowsInHeader; i++) {
        header[i] = (char *)malloc(maxSizeHeadRow);
        if (!header[i]) {
            perror("malloc header");
            fclose(in); fclose(out);
            // free already allocated
            for (int j = 0; j < i; j++) free(header[j]);
            return 1;
        }
    }

    int rows = 0, cols = 0;
    int **pixels = seqPgmRead(header, &rows, &cols, in);
    if (!pixels) {
        fprintf(stderr, "seqPgmRead failed\n");
        freeHeader(header);
        fclose(in); fclose(out);
        return 1;
    }

    int rc = 0;
    switch (op) {
        case OP_CIRCLE:
            rc = seqPgmDrawCircle(pixels, rows, cols, centerRow, centerCol, radius, header);
            break;
        case OP_EDGE:
            rc = seqPgmDrawEdge(pixels, rows, cols, edgeWidth, header);
            break;
        case OP_LINE:
            rc = seqPgmDrawLine(pixels, rows, cols, header, p1row, p1col, p2row, p2col);
            break;
        default:
            rc = 1;
            break;
    }

    if (rc != 0) {
        fprintf(stderr, "Draw operation failed\n");
        freePixels(pixels, rows);
        freeHeader(header);
        fclose(in); fclose(out);
        return 1;
    }

    if (seqPgmWrite((const char **)header, (const int **)pixels, rows, cols, out) != 0) {
        fprintf(stderr, "seqPgmWrite failed\n");
        freePixels(pixels, rows);
        freeHeader(header);
        fclose(in); fclose(out);
        return 1;
    }

    freePixels(pixels, rows);
    freeHeader(header);
    fclose(in);
    fclose(out);
    return 0;
}


// Instructions for compiling and running:
/*
Compile:
    gcc -Wall -Wextra -O2 -std=c11 seqMain.c seqPgmUtility.c -o seq.exe -lm
Circle:
    .\seq.exe -c 256 256 100 baboon.ascii.pgm baboon_c100_seq.pgm
Edge:
    .\seq.exe -e 50 baboon.ascii.pgm baboon_e50_seq.pgm
Line:
    .\seq.exe -l 1 5 300 300 baboon.ascii.pgm baboon_l1_seq.pgm
*/