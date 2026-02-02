#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"

int main() {

    char[] flag = argv[1];
    switch (flag) {
        case "-c":
            pgmDrawCircle(pixels, numRows, numCols, centerRow, centerCol, radius, header);
            break;
        case "edge":
            pgmDrawEdge(pixels, numRows, numCols, edgeWidth, header);
            break;
        case "line":
            pgmDrawLine(pixels, numRows, numCols, header, p1row, p1col, p2row, p2col);
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