//
//  pgmUtility.c
//  cscd240PGM
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "seqPgmUtility.h"

//---------------------------------------------------------------------------
int ** seqPgmRead( char **header, int *numRows, int *numCols, FILE *in )
{
    int i, j;
    
    // read in header of the image first
    for( i = 0; i < rowsInHeader; i ++)
    {
        if ( header[i] == NULL )
        {
            return NULL;
        }
        if( fgets( header[i], maxSizeHeadRow, in ) == NULL )
        {
            return NULL;
        }
    }
    // extract rows of pixels and columns of pixels
    sscanf( header[rowsInHeader - 2], "%d %d", numCols, numRows );  // in pgm the first number is # of cols
    
    // Now we can intialize the pixel of 2D array, allocating memory
    int **pixels = malloc((*numRows) * sizeof(int*));
    if (pixels == NULL) {
        return NULL;
    }

    for (i = 0; i < *numRows; i++) {
        pixels[i] = malloc((*numCols) * sizeof(int));
        if (pixels[i] == NULL) {
            // clean up already allocated rows
            for (j = 0; j < i; j++) {
                free(pixels[j]);
            }
            free(pixels);
            return NULL;
        }
    }

    
    // read in all pixels into the pixels array.
    for( i = 0; i < *numRows; i ++ )
        for( j = 0; j < *numCols; j ++ )
            if ( fscanf(in, "%d ", *( pixels + i ) + j) < 0 )
                return NULL;
    
    return pixels;
}

//---------------------------------------------------------------------------
//
int seqPgmDrawCircle( int **pixels, int numRows, int numCols, int centerRow, int centerCol, int radius, char **header ) {
    // 
    (void)header;

    // Iterate over each pixel
    for(int r = 0; r < numRows; r++) {
        for(int c = 0; c < numCols; c++) {
            // Calculate distance from center
            double dr = (double)(r - centerRow);
            double dc = (double)(c - centerCol);
            double distance = sqrt(dr * dr + dc * dc);
            // If within the circle, set pixel to zero
            if (distance <= radius)
 {
                pixels[r][c] = 0;
                //Black pixel
            }
        }
    }
    return 0;
}

//---------------------------------------------------------------------------
int seqPgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header ) {
    (void)header;

    // Clamp negative edgeWidth to zero
    if(edgeWidth <= 0) edgeWidth = 0;

    for(int r = 0; r < numRows; r++) {
        for(int c = 0; c < numCols; c++) {
            // Check if pixel is within edgeWidth of any border
            if(r < edgeWidth || r >= numRows - edgeWidth ||
               c < edgeWidth || c >= numCols - edgeWidth) {
                pixels[r][c] = 0;
                //Black pixel 
            }
        }
    }

    return 0;
}

//---------------------------------------------------------------------------
static double pointToSegmentDistance( double px, double py, double ax, double ay, double bx, double by) {
    // Vector AB
    double abx = bx - ax;
    double aby = by - ay;
    // Vector AP
    double apx = px - ax;
    double apy = py - ay;
    // Length squared of AB
    double abLen2 = abx * abx + aby * aby;
    if(abLen2 == 0.0) {
        // A and B are the same point
        double dx = px - ax;
        double dy = py - ay;
        return sqrt(dx * dx + dy * dy);
    }

    // Project point AP onto line AB
    double t = (apx * abx + apy * aby) / abLen2;
    if(t < 0.0) t = 0.0;
    else if(t > 1.0) t = 1.0;

    // Find the closest point C on segment AB
    double cx = ax + t * abx;
    double cy = ay + t * aby;
    // Distance from P to C
    double dx = px - cx;
    double dy = py - cy;
    return sqrt(dx * dx + dy * dy);
}

int seqPgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    (void)header;

    //Converts to Cartesian coordinates
    double ax = (double)p1col, ay = (double)p1row;
    double bx = (double)p2col, by = (double)p2row;

    // Iterate over each pixel
    for(int r = 0; r < numRows; r++) {
        for(int c =0; c < numCols; c++) {
            double px = (double)c;
            double py = (double)r;

            // Calculate distance from pixel center to line segment
            double d = pointToSegmentDistance(px, py, ax, ay, bx, by);

            if(d <= 0.5) {
                pixels[r][c] = 0;
                //Black pixel
            }
        }
    }
    return 0;
}

//----------------------------------------------------------------------------
int seqPgmWrite( const char **header, const int **pixels, int numRows, int numCols, FILE *out ) {
    int i, j;
    
    // write the header
    for ( i = 0; i < rowsInHeader; i ++ )
    {
        fprintf(out, "%s", *( header + i ) );
    }
    
    // write the pixels
    for( i = 0; i < numRows; i ++ )
    {
        for ( j = 0; j < numCols; j ++ )
        {
            if ( j < numCols - 1 )
                fprintf(out, "%d ", pixels[i][j]);
            else
                fprintf(out, "%d\n", pixels[i][j]);
        }
    }
    return 0;
}

//-------------------------------------------------------------------------------
double seqDistance( int p1[], int p2[] )
{
    return sqrt( pow( p1[0] - p2[0], 2 ) + pow( p1[1] - p2[1], 2 ) );
}


