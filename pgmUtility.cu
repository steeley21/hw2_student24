#include <stdio.h>
#include <stdlib.h>
#include "pgmUtility.h"


// // Implement or define each function prototypes listed in pgmUtility.h file.
// // NOTE: Please follow the instructions stated in the write-up regarding the interface of the functions.
// // NOTE: You might have to change the name of this file into pgmUtility.cu if needed.

int * pgmRead( char **header, int *numRows, int *numCols, FILE *in ) {
    int i, j;
    
    // read in header of the image first
    for( i = 0; i < rowsInHeader; i ++) {
        if ( header[i] == NULL ) {
            return NULL;
        }
        if( fgets( header[i], maxSizeHeadRow, in ) == NULL ) {
            return NULL;
        }
    }
    // extract rows of pixels and columns of pixels
    sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );  // in pgm the first number is # of cols
    
    // Now we can intialize the pixel of 1D Array, allocating memory
    int *pixels = malloc((*numRows) * (*numCols) * sizeof(int));
    if ( pixels == NULL ) {
        return NULL;
    }
    for (i = 0; i < *numRows; i ++) {
        for (j = 0; j < *numCols; j++) {
            *(pixels + i * (*numCols) + j) = 0;
        }
    }
    }
    
    // read in all pixels into the pixels array.
    for(i = 0; i < *numRows; i++ ) {
        for(j = 0; j < *numCols; j++) {
            if (fscanf(in, "%d ", &pixels[i * (*numCols) + j]) < 0) {
                return NULL;
            }
        }
    }
    
    return pixels;
}

//---------------------------------------------------------------------------
//
int pgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header ) {
    
    drawCircleKernel<<<1,1>>>(pixels, numRows, numCols, centerRow, centerCol, radius);
    cudaDeviceSynchronize();
    return 0;
}

//---------------------------------------------------------------------------
int pgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header )
{
    int *d_pixels;
    int size = numRows * numCols * sizeof(int);

    cudaMalloc((void**)&d_pixels, size); // Allocate device memory
    cudaMemcpy(d_pixels, pixels, size, cudaMemcpyHostToDevice); // Copy pixels to device

    dim3 block(16, 16); // Define block size
    dim3 grid((numCols + block.x - 1) / block.x, (numRows + block.y - 1) / block.y); // Define grid size

    pgmDrawEdgeKernel<<<grid, block>>>(d_pixels, numRows, numCols, edgeWidth); // Launch kernel
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost); // Copy result back to host
    cudaFree(d_pixels); // Free device memory
    return 0;
}

//---------------------------------------------------------------------------

int pgmDrawLine(int* pixels, int numRows, int numCols, char** header,
                int p1row, int p1col, int p2row, int p2col) {
    int total = numRows * numCols;
    size_t bytes = sizeof(int) * (size_t)total;

    int* d_pixels = nullptr;
    dieCuda(cudaMalloc((void**)&d_pixels, bytes), "cudaMalloc d_pixels">
    dieCuda(cudaMemcpy(d_pixels, pixels, bytes, cudaMemcpyHostToDevice)>

    int block = 256;
    int grid = (total + block - 1) / block;
    lineKernel<<<grid, block>>>(d_pixels, numRows, numCols, p1row, p1co>
    dieCuda(cudaGetLastError(), "launch lineKernel");
    dieCuda(cudaDeviceSynchronize(), "sync lineKernel");

    dieCuda(cudaMemcpy(pixels, d_pixels, bytes, cudaMemcpyDeviceToHost)>
    cudaFree(d_pixels);

    int oldMax = parseMaxFromHeader(header);
    int newMax = computeMaxCPU(pixels, total);
    if (newMax != oldMax) {
        writeMaxToHeader(header, newMax);
        return 1;
    }
    return 0;
}

//----------------------------------------------------------------------------
int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out )
{
    int i, j;
    
    // write the header
    for ( i = 0; i < rowsInHeader; i ++ )
    {
        fprintf(out, "%s", *( header + i ) );
    }
    
    // write the pixels
    for( i = 0; i < numRows; i ++ ) {
        for ( j = 0; j < numCols; j ++ ) {
            if ( j < numCols - 1 )
                fprintf(out, "%d ", pixels[i * numCols + j]);
            else
                fprintf(out, "%d\n", pixels[i * numCols + j]);
        }
    }
    return 0;
}

//-------------------------------------------------------------------------------
double distance( int p1[], int p2[] )
{
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}