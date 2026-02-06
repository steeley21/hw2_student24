#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s cols rows output.pgm\n", argv[0]);
        return 1;
    }
    int cols = atoi(argv[1]);
    int rows = atoi(argv[2]);
    const char *out = argv[3];

    if (cols <= 0 || rows <= 0) {
        fprintf(stderr, "cols/rows must be positive\n");
        return 1;
    }

    FILE *f = fopen(out, "w");
    if (!f) { perror("fopen"); return 1; }

    fprintf(f, "P2\n");
    fprintf(f, "# synthesized %dx%d all-255\n", cols, rows);
    fprintf(f, "%d %d\n", cols, rows);
    fprintf(f, "255\n");

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            fputs("255", f);
            if (c < cols - 1) fputc(' ', f);
        }
        fputc('\n', f);
    }

    fclose(f);
    return 0;
}

/*
Compile:
    gcc -O2 gen_pgm.c -o gen_pgm
Generate:
    ./gen_pgm 10000 10000 white_10000x10000.pgm
    ./gen_pgm 10000 1000  white_1000x10000.pgm
*/