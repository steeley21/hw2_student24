

/**
 *  Function Name:
 *      distance()
 *      distance() returns the Euclidean distance between two pixels. This function is executed on CUDA device
 *
 *  @param[in]  p1  coordinates of pixel one, p1[0] is for row number, p1[1] is for column number
 *  @param[in]  p2  coordinates of pixel two, p2[0] is for row number, p2[1] is for column number
 *  @return         return distance between p1 and p2
 */
__device__ float distance(int p1[], int p2[]) {
    return sqrtf( (float)( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) ) );
}

__global__ void drawCircleKernel(int *pixels, int numRows, int numCols, int centerRow, int centerCol, int radius) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < numRows && col < numCols) {
        int p[2] = {row, col};
        int center[2] = {centerRow, centerCol};
        float dist = distance(p, center);
        if (dist <= radius) {
            pixels[row * numCols + col] = 0;
        }
    }

}

__global__ void drawEdgeKernel(int *pixels, int numRows, int numCols, int edgeWidth) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < numRows && col < numCols) {
        if (row < edgeWidth || row >= numRows - edgeWidth || col < edgeWidth || col >= numCols - edgeWidth) {
            pixels[row * numCols + col] = 0; // Set pixel to black if it's within the edge width
        }
    }

}

__global__ void drawLineKernel(int *pixels, int numRows, int numCols, int p1row, int p1col, int p2row, int p2col) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < numRows && col < numCols) {
        // Calculate line equation parameters
        float A = (float)(p2row - p1row);
        float B = (float)(p1col - p2col);
        float C = (float)(p2col * p1row - p1col * p2row);

        // Calculate distance from point to line
        float dist = fabs(A * row + B * col + C) / sqrtf(A * A + B * B);
        if (dist <= 0.5) {
            float dot = (col - p1col) * (p2col - p1col) + (row - p1row) * (p2row - p1row);

            float len2 = (p2col - p1col) * (p2col - p1col) + (p2row - p1row) * (p2row - p1row);

            if (dot >= 0 && dot <= len2) {
                pixels[row * numCols + col] = 0;
            }
        }
    }

}