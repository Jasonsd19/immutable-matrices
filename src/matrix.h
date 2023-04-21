#ifndef LIBMATRIX_H
#define LIBMATRIX_H

#include <vector>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <math.h>
#include <thread>

typedef std::vector<double>              doubleArray_t;
typedef std::vector<doubleArray_t>       twoDimDoubleArray_t;

class matrix {

private:
    doubleArray_t mData;

    const unsigned int rows;

    const unsigned int columns;

public:
    matrix(doubleArray_t data, int rows, int columns);

    matrix(doubleArray_t data, int n);

    matrix(twoDimDoubleArray_t);

    ~matrix();

    doubleArray_t getData();

    unsigned int getRows();

    unsigned int getColumns();

    matrix operator+(matrix m);

    matrix operator-(matrix m);

    matrix operator*(matrix m);

    matrix operator/(matrix m);

    double operator()(unsigned int i, unsigned int j);

    friend std::ostream& operator<<(std::ostream& os, matrix& m);

    static bool sameDims(matrix M, matrix M2);

    static matrix scalarMultiply(matrix M, double scalar);

    static matrix matrixMultiply(matrix leftMatrix, matrix rightMatrix);

    static matrix transpose(matrix M);

    static matrix identityMatrix(int n);

    static std::tuple<matrix, matrix, matrix, int> LUPDecompose(matrix M);

    static matrix solve(matrix M, matrix vector);

    static matrix solveLUP(matrix L, matrix U, matrix P, int swaps, matrix vector);

    static double determinant(matrix M);

    static double determinantLUP(matrix L, matrix U, int swaps);

    static matrix inverse(matrix M);

    static matrix inverseLUP(matrix L, matrix U, matrix P, int swaps);
};

#endif