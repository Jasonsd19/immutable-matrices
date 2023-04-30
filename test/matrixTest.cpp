#include <iostream>
#include <string>
#include <sstream> 
#include <cstring>
#include <matrix.h>
#include <typeinfo>
#include <math.h>

#define isTrue(x, y) { if (!(x)) { std::stringstream ss; ss << "  - " << __func__ << " assertation failed on line " << __LINE__ << "\n"; y.push_back(ss.str());} }

// Error threshold for double comparisons
const double EPSILON = 2.2e-14;

// Prints results of test to standard output
void outputTestResult(std::vector<std::string>* failedTests, const char* func) {
    if (failedTests->size() == 0) {
        std::cout << "✅ " << "All tests passed for " << func << std::endl;
    }
    else {
        std::cout << "❌ " << func << " failed at:" << std::endl;
        for (auto i = 0; i < failedTests->size(); i++) {
            std::cout << (*failedTests)[i];
        }
    }
}

// Compares two doubles and returns true if they are within the error threshold of each other
bool isDoubleSame(double d1, double d2) {
    double diff = fabs(d1 - d2);
    d1 = fabs(d1);
    d2 = fabs(d2);
    double largest = d1 > d2 ? d1 : d2;

    if (diff <= largest * EPSILON) {
        return true;
    }

    return false;
}

// Compares two vectors<double> and returns true if they contain the same elements
bool isDataSame(doubleArray_t array1, doubleArray_t array2) {
    if (array1.size() != array2.size()) {
        return false;
    }

    for (auto i = 0; i < array1.size(); i++) {
        if (!isDoubleSame(array1[i], array2[i])) {
            return false;
        }
    }

    return true;
}

void testMatrixConstructor() {
    std::vector<std::string> failedTests;

    doubleArray_t squareData = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    doubleArray_t squareData2 = { 1, 2, 3, 4 };
    doubleArray_t nonSquareData = { 1, 2, 3, 4, 5, 6, 7, 8 };
    twoDimDoubleArray_t twoDimData = { {1, 2, 3}, {4, 5, 6} };

    matrix square = matrix(squareData, 3);
    matrix nonSquare = matrix(nonSquareData, 4, 2);
    matrix twoDim = matrix(twoDimData);
    matrix truncatedData = matrix(squareData, 2);
    matrix overflowData = matrix(squareData2, 3, 3);

    isTrue(square.getRows() == 3 && square.getColumns() == 3, failedTests);
    isTrue(nonSquare.getRows() == 4 && nonSquare.getColumns() == 2, failedTests);
    isTrue(twoDim.getRows() == 2 && twoDim.getColumns() == 3, failedTests);
    isTrue(truncatedData.getRows() == 2 && truncatedData.getColumns() == 2, failedTests);
    isTrue(overflowData.getRows() == 3 && overflowData.getColumns() == 3, failedTests);

    isTrue(isDataSame(square.getData(), squareData), failedTests);
    isTrue(isDataSame(nonSquare.getData(), nonSquareData), failedTests);
    isTrue(isDataSame(twoDim.getData(), doubleArray_t({ 1, 2, 3 , 4, 5, 6 })), failedTests);
    isTrue(isDataSame(truncatedData.getData(), doubleArray_t({ 1, 2, 3, 4 })), failedTests);
    isTrue(isDataSame(overflowData.getData(), doubleArray_t({ 1, 2, 3, 4, 0, 0, 0, 0, 0 })), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testMatrixGetters() {
    std::vector<std::string> failedTests;

    doubleArray_t nonSquareData = { 1, 2, 3, 4, 5, 6 };
    matrix nonSquare = matrix(nonSquareData, 3, 2);

    isTrue(isDataSame(nonSquare.getData(), nonSquareData), failedTests);
    isTrue(nonSquare.getColumns() == 2, failedTests);
    isTrue(nonSquare.getRows() == 3, failedTests);

    outputTestResult(&failedTests, __func__);
}

void testMatrixOperators() {
    std::vector<std::string> failedTests;

    doubleArray_t squareData = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    doubleArray_t nonSquareData = { 1, 2, 3, 4, 5, 6 };
    doubleArray_t nonSquareData2 = { 6, 5, 4, 3, 2, 1 };

    matrix square = matrix(squareData, 3);
    matrix nonSquare = matrix(nonSquareData, 3, 2);
    matrix nonSquare2 = matrix(nonSquareData2, 3, 2);

    matrix addition = nonSquare + nonSquare2;
    matrix subtraction = nonSquare - nonSquare2;
    matrix multiplication = nonSquare * nonSquare2;
    matrix division = nonSquare / nonSquare;

    isTrue(isDataSame(addition.getData(), doubleArray_t({ 7, 7, 7, 7, 7, 7 })), failedTests);
    isTrue(isDataSame(subtraction.getData(), doubleArray_t({ -5, -3, -1, 1, 3, 5 })), failedTests);
    isTrue(isDataSame(multiplication.getData(), doubleArray_t({ 6, 10, 12, 12, 10, 6 })), failedTests);
    isTrue(isDataSame(division.getData(), doubleArray_t({ 1, 1, 1, 1, 1, 1 })), failedTests);

    try {
        matrix failError = square + nonSquare;
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Mismatched dimensions between matrices.") == 0, failedTests);
    }

    try {
        matrix failError = square - nonSquare;
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Mismatched dimensions between matrices.") == 0, failedTests);
    }

    try {
        matrix failError = square * nonSquare;
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Mismatched dimensions between matrices.") == 0, failedTests);
    }

    try {
        matrix failError = square / nonSquare;
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Mismatched dimensions between matrices.") == 0, failedTests);
    }


    outputTestResult(&failedTests, __func__);
}

void testSameDims() {
    std::vector<std::string> failedTests;

    doubleArray_t squareData = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    doubleArray_t nonSquareData = { 1, 2, 3, 4, 5, 6 };

    matrix square = matrix(squareData, 3);
    matrix nonSquare = matrix(nonSquareData, 3, 2);

    isTrue(matrix::sameDims(square, square), failedTests);
    isTrue(!matrix::sameDims(square, nonSquare), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testScalarMultiply() {
    std::vector<std::string> failedTests;

    doubleArray_t squareData = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix square = matrix(squareData, 3);

    isTrue(isDataSame(matrix::scalarMultiply(square, 2.5).getData(), doubleArray_t({ 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5 })), failedTests);
    isTrue(isDataSame(matrix::scalarMultiply(square, 2).getData(), doubleArray_t({ 2, 4, 6, 8, 10, 12, 14, 16, 18 })), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testMatrixMultiply() {
    std::vector<std::string> failedTests;

    doubleArray_t data1 = { 1, 5, 3, 2, 4, 2, 3, 5, 1, 5, 9, 8 };
    doubleArray_t data2 = { 2, 5, 4, 8, 7, 3, 2, 5 };

    matrix m1 = matrix(data1, 3, 4);
    matrix m2 = matrix(data2, 4, 2);

    isTrue(isDataSame(matrix::matrixMultiply(m1, m2).getData(), doubleArray_t({ 47, 64, 47, 70, 101, 112 })), failedTests);

    try {
        matrix failError = matrix::matrixMultiply(m1, m1);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "The number of columns of the left matrix must equal the number of rows of the right matrix.") == 0, failedTests);
    }



    outputTestResult(&failedTests, __func__);
}

void testTranspose() {
    std::vector<std::string> failedTests;

    doubleArray_t data1 = { 1, 5, 3, 2, 4, 2, 3, 5, 1, 5, 9, 8 };
    doubleArray_t data2 = { 2, 5, 4 };

    matrix m1 = matrix(data1, 3, 4);
    matrix m2 = matrix(data2, 3, 1);

    isTrue(isDataSame(matrix::transpose(m1).getData(), doubleArray_t({ 1, 4, 1, 5, 2, 5, 3, 3, 9, 2, 5, 8 })), failedTests);
    isTrue(isDataSame(matrix::transpose(m2).getData(), data2), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testIdentityMatrix() {
    std::vector<std::string> failedTests;

    matrix identity = matrix::identityMatrix(3);

    isTrue(isDataSame(identity.getData(), doubleArray_t({ 1, 0, 0, 0, 1, 0, 0, 0, 1 })), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testLUPDecompose() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { 2, 4, 2, 4, -10, 2, 1, 2, 4 };
    doubleArray_t m2Data = { 2, 4, 2, 4, -10, 2 };

    matrix m1 = matrix(m1Data, 3);
    matrix m2 = matrix(m2Data, 3, 2);

    auto [P, U, L, swaps] = matrix::LUPDecompose(m1);

    isTrue(L.getData() == doubleArray_t({ 1, 0, 0, 0.5, 1, 0, 0.25, 0.5, 1 }), failedTests);
    isTrue(U.getData() == doubleArray_t({ 4, -10, 2, 0, 9, 1, 0, 0, 3 }), failedTests);
    isTrue(P.getData() == doubleArray_t({ 0, 1, 0, 1, 0, 0, 0, 0, 1 }), failedTests);

    try {
        auto [pFail, uFail, lFail, swapsFail] = matrix::LUPDecompose(m2);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "The matrix must be square.") == 0, failedTests);
    }

    outputTestResult(&failedTests, __func__);
}

void testSolveLUP() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { 5, 8, -4, 6, 9, -5, 4, 7, -2 };
    doubleArray_t m2Data = { 6, 9, 0, 4, 2, 0, 6, 9, 0 };

    matrix m1 = matrix(m1Data, 3);
    matrix m2 = matrix(m2Data, 3);

    auto [P, U, L, swaps] = matrix::LUPDecompose(m1);
    auto [P2, U2, L2, swaps2] = matrix::LUPDecompose(m2);


    matrix m1Solution = matrix::solveLUP(L, U, P, swaps, matrix(doubleArray_t({ -18, -20, -15 }), 3, 1));

    isTrue(isDataSame(m1Solution.getData(), doubleArray_t({ 2, -3, 1 })), failedTests);

    try {
        matrix m2SolutionFail = matrix::solveLUP(L2, U2, P2, swaps2, matrix(doubleArray_t({ -18, -20, -15 }), 3, 1));
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "There is no unique solution.") == 0, failedTests);
    }

    outputTestResult(&failedTests, __func__);
}

void testSolve() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { 5, 8, -4, 6, 9, -5, 4, 7, -2 };
    doubleArray_t m2Data = { 6, 9, 0, 4, 2, 0, 6, 9, 0 };

    matrix m1 = matrix(m1Data, 3);
    matrix m2 = matrix(m2Data, 3);

    matrix m1Solution = matrix::solve(m1, matrix(doubleArray_t({ -18, -20, -15 }), 3, 1));

    isTrue(isDataSame(m1Solution.getData(), doubleArray_t({ 2, -3, 1 })), failedTests);

    try {
        matrix m2SolutionFail = matrix::solve(m2, matrix(doubleArray_t({ -18, -20, -15 }), 3, 1));
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "There is no unique solution.") == 0, failedTests);
    }

    outputTestResult(&failedTests, __func__);
}

void testDeterminantLUP() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { 1, 0, 5, 2, 1, 6, 2, 2, 4 };
    matrix m1 = matrix(m1Data, 3);

    auto [P, U, L, swaps] = matrix::LUPDecompose(m1);

    isTrue(isDoubleSame(matrix::determinantLUP(L, U, swaps), 2.0), failedTests);

    try {
        matrix failMatrix = matrix(doubleArray_t({ 1, 0, 5, 2, 9, 8 }), 3, 2);
        double determinantFail = isDoubleSame(matrix::determinantLUP(L, failMatrix, swaps), 2);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "One or more of the matrices provided are not square.") == 0, failedTests);
    }

    try {
        matrix failMatrix = matrix(doubleArray_t({ 1, 0, 5, 2 }), 2, 2);
        double determinantFail = isDoubleSame(matrix::determinantLUP(L, failMatrix, swaps), 2);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Lower, and Upper matrices must have the same dimensions.") == 0, failedTests);
    }

    outputTestResult(&failedTests, __func__);
}

void testDeterminant() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { 1, 0, 5, 2, 1, 6, 2, 2, 4 };
    matrix m1 = matrix(m1Data, 3);

    isTrue(isDoubleSame(matrix::determinant(m1), 2.0), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testInverseLUP() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    doubleArray_t results = { -6, 7, 2, 4, -5, -1, (5.0 / 3.0), -2, -(2.0 / 3.0) };

    matrix m1 = matrix(m1Data, 3);

    auto [P, U, L, swaps] = matrix::LUPDecompose(m1);

    matrix inverse = matrix::inverseLUP(L, U, P, swaps);

    isTrue(isDataSame(inverse.getData(), results), failedTests);

    try {
        matrix failMatrix = matrix(doubleArray_t({ 1, 0, 5, 2, 1, 6 }), 3, 2);
        double inverseFail = isDataSame(matrix::inverseLUP(L, failMatrix, P, swaps).getData(), results);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "One or more of the matrices provided are not square.") == 0, failedTests);
    }

    try {
        matrix failMatrix = matrix(doubleArray_t({ 1, 0, 5, 2 }), 2);
        double inverseFail = isDataSame(matrix::inverseLUP(L, failMatrix, P, swaps).getData(), results);
        isTrue(false, failedTests);
    }
    catch (const std::exception& e) {
        isTrue(strcmp(e.what(), "Lower, Upper, and Permutation matrices must have the same dimensions.") == 0, failedTests);
    }

    outputTestResult(&failedTests, __func__);
}

void testInverse() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    doubleArray_t results = { -6, 7, 2, 4, -5, -1, (5.0 / 3.0), -2, -(2.0 / 3.0) };

    matrix m1 = matrix(m1Data, 3);

    isTrue(isDataSame(matrix::inverse(m1).getData(), results), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testGetColumn() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    doubleArray_t results = { -4, -3,  -1 };

    matrix m1 = matrix(m1Data, 3);

    isTrue(isDataSame(matrix::getColumn(m1, 0).getData(), results), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testGetRow() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    doubleArray_t results = { -4, -2, -9 };

    matrix m1 = matrix(m1Data, 3);

    isTrue(isDataSame(matrix::getRow(m1, 0).getData(), results), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testSum() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    double results = -32;

    matrix m1 = matrix(m1Data, 3);

    isTrue(matrix::sum(m1) == results, failedTests);

    outputTestResult(&failedTests, __func__);
}

void testMap() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    doubleArray_t results = { 4, 2, 9, 3, 2, 6, 1, -1, 6 };

    matrix m1 = matrix(m1Data, 3);

    isTrue(isDataSame(matrix::map(m1, [](double x) { return x * -1; }).getData(), results), failedTests);

    outputTestResult(&failedTests, __func__);
}

void testMapSum() {
    std::vector<std::string> failedTests;

    doubleArray_t m1Data = { -4, -2, -9, -3, -2, -6, -1, 1, -6 };
    double results = 64;

    matrix m1 = matrix(m1Data, 3);
    matrix mappedM1 = matrix::map(m1, [](double x) { return x * -2; });

    isTrue(matrix::sum(mappedM1) == results, failedTests);

    outputTestResult(&failedTests, __func__);
}

// Call all test functions
int main() {
    testMatrixConstructor();
    testMatrixGetters();
    testMatrixOperators();
    testSameDims();
    testScalarMultiply();
    testMatrixMultiply();
    testTranspose();
    testIdentityMatrix();
    testLUPDecompose();
    testSolveLUP();
    testSolve();
    testDeterminantLUP();
    testDeterminant();
    testInverseLUP();
    testInverse();
    testGetColumn();
    testGetRow();
    testSum();
    testMap();
    testMapSum();
}