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

int pgmDrawCircle( int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header ) {

    int *device_pixels;
    int byteSize = numRows * numCols * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&device_pixels, byteSize); 
    cudaMemcpy(device_pixels, pixels, byteSize, cudaMemcpyHostToDevice);
    
    //Calculate Parameters
    dim3 block(16, 16); // Define block size
    dim3 grid((numCols + block.x - 1) / block.x, (numRows + block.y - 1) / block.y);

    //Launch Kernel
    drawCircleKernel<<<grid, block>>>(device_pixels, numRows, numCols, centerRow, centerCol, radius);

    // Copy result back to host and free device memory
    cudaMemcpy(pixels, device_pixels, size, cudaMemcpyDeviceToHost); 
    cudaFree(device_pixels);

    return 0;
}

int pgmDrawEdge(int *pixels, int numRows, int numCols, int edgeWidth, char **header)
{
    int *device_pixels;
    int byteSize = numRows * numCols * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&device_pixels, byteSize);
    cudaMemcpy(device_pixels, pixels, byteSize, cudaMemcpyHostToDevice);
    
    //Calculate Parameters
    dim3 block(16, 16); 
    dim3 grid((numCols + block.x - 1) / block.x, (numRows + block.y - 1) / block.y); 

    //Launch Kernel
    pgmDrawEdgeKernel<<<grid, block>>>(device_pixels, numRows, numCols, edgeWidth); 
    cudaDeviceSynchronize();

    //Copy result back to host and free device memory
    cudaMemcpy(pixels, device_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(device_pixels); 
    return 0;
}

/*
/ pgmDrawLine NEEDS WORK
*/
int pgmDrawLine(int* pixels, int numRows, int numCols, char** header, int p1row, int p1col, int p2row, int p2col) {

    int *device_pixels;
    int byteSize = numRows * numCols * sizeof(int);

    int* device_pixels = nullptr;
    dieCuda(cudaMalloc((void**)&device_pixels, bytes), "cudaMalloc device_pixels");
    dieCuda(cudaMemcpy(device_pixels, pixels, bytes, cudaMemcpyHostToDevice), "cudaMemcpy device_pixels");
    int block = 256;
    int grid = (total + block - 1) / block;
    lineKernel<<<grid, block>>>(device_pixels, numRows, numCols, p1row, p1col, p2row, p2col);
    dieCuda(cudaGetLastError(), "launch lineKernel");
    dieCuda(cudaDeviceSynchronize(), "sync lineKernel");

    dieCuda(cudaMemcpy(pixels, device_pixels, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy pixels");
    cudaFree(device_pixels);

    return 0;
}

int pgmWrite( const char **header, const int *pixels, int numRows, int numCols, FILE *out )  {
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