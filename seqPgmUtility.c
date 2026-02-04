//
//  pgmUtility.c
//  cscd240PGM
//
//  Created by Tony Tian on 11/2/13.
//  Copyright (c) 2013 Tony Tian. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>

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
    int **pixels = ( int ** ) malloc( ( *numRows ) * sizeof( int * ) );
    for( i = 0; i < *numRows; i ++)
    {
        pixels[i] = ( int * ) malloc( ( *numCols ) * sizeof( int ) );
        if ( pixels[i] == NULL )
        {
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
    return 0;
}

//---------------------------------------------------------------------------
int seqPgmDrawEdge( int **pixels, int numRows, int numCols, int edgeWidth, char **header ) {
    return 0;
}

//---------------------------------------------------------------------------
static double pointToSegmentDistance( double px, double py, double ax, double ay, double bx, double by) {
    double abx = bx - ax;
    double aby = by - ay;
    double apx = px - ax;
    double apy = py - ay;

    double abLen2 = abx * abx + aby * aby;
    if(abLen2 == 0.0) {
        // A and B are the same point
        double dx = px - ax;
        double dy = py - ay;
        return sqrt(dx * dx + dy * dy);
    }

    double t = (apx * abx + apy * aby) / abLen2;
    if(t < 0.0) t = 0.0;
    else if(t > 1.0) t = 1.0;

    double cx = ax + t * abx;
    double cy = ay + t * aby;
    double dx = px - cx;
    double dy = py - cy;
    return sqrt(dx * dx + dy * dy);
}

int seqPgmDrawLine( int **pixels, int numRows, int numCols, char **header, int p1row, int p1col, int p2row, int p2col ) {
    (void)header;

    double ax = (double)p1col, ay = (double)p1row;
    double bx = (double)p2col, by = (double)p2row;

    for(int r = 0; r < numRows; r++) {
        for(int c =0; c < numCols; c++) {
            double px = (double)c;
            double py = (double)r;

            double d = pointToSegmentDistance(px, py, ax, ay, bx, by);

            if(d <= 0.5) {
                pixels[r][c] = 0;
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


