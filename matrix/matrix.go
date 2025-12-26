package matrix

import (
	"fmt"
	"sync"
)

// Matrix type is an alias for a 2D slice of float64
type Matrix [][]float64

// NewMatrix creates and returns a new Matrix with the given number of rows and columns.
// All elements are initialized to 0.0.
func NewMatrix(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// ScalarMultiply multiplies each element of the matrix by a scalar value.
func (m Matrix) ScalarMultiply(scalar float64) Matrix {
	rows := len(m)
	cols := len(m[0])
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

// MultiplyElementWise performs element-wise multiplication of two matrices.
func (a Matrix) MultiplyElementWise(b Matrix) (Matrix, error) {
	rowsA := len(a)
	colsA := len(a[0])
	rowsB := len(b)
	colsB := len(b[0])

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("incompatible dimensions for element-wise multiplication: %dx%d and %dx%d", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsA)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result, nil
}

// Transpose returns a new matrix that is the transpose of the current matrix.
func (a Matrix) Transpose() Matrix {
	rows := len(a)
	cols := 0
	if rows > 0 {
		cols = len(a[0])
	}
	

	result := NewMatrix(cols, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = a[i][j]
		}
	}
	return result
}

// Apply applies a function to each element of the matrix, returning a new matrix with the results.
func (a Matrix) Apply(fn func(float64) float64) Matrix {
	rows := len(a)
	cols := len(a[0])
	result := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = fn(a[i][j])
		}
	}
	return result
}

// Subtract performs element-wise subtraction between the current matrix and another matrix (b).
// It returns a new matrix with the result or an error if dimensions are incompatible.
func (a Matrix) Subtract(b Matrix) (Matrix, error) {
	rowsA := len(a)
	colsA := len(a[0])
	rowsB := len(b)
	colsB := len(b[0])

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("incompatible dimensions for subtraction: %dx%d and %dx%d", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsA)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result, nil
}

// Add performs element-wise addition between the current matrix and another matrix (b).
// It returns a new matrix with the result or an error if dimensions are incompatible.
func (a Matrix) Add(b Matrix) (Matrix, error) {
	rowsA := len(a)
	colsA := len(a[0])
	rowsB := len(b)
	colsB := len(b[0])

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("incompatible dimensions for addition: %dx%d and %dx%d", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsA)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result, nil
}

// DotProduct performs matrix multiplication between the current matrix and another matrix (B).
// It returns a new matrix with the result or an error if dimensions are incompatible.
func (a Matrix) DotProduct(b Matrix) (Matrix, error) {
	rowsA := len(a)
	colsA := 0
	if rowsA > 0 {
		colsA = len(a[0])
	}
	rowsB := len(b)
	colsB := 0
	if rowsB > 0 {
		colsB = len(b[0])
	}

	if colsA != rowsB {
		return nil, fmt.Errorf("incompatible dimensions for dot product: %dx%d and %dx%d", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsB)
	
	// Adaptive Parallelism Threshold
	// For small batch sizes (like single inference), the overhead of goroutines outweighs the benefits.
	const parallelThreshold = 4

	if rowsA < parallelThreshold {
		// Sequential execution for small matrices
		for i := 0; i < rowsA; i++ {
			for j := 0; j < colsB; j++ {
				sum := 0.0
				for k := 0; k < colsA; k++ {
					sum += a[i][k] * b[k][j]
				}
				result[i][j] = sum
			}
		}
	} else {
		// Parallel execution for large matrices (Training batches)
		var wg sync.WaitGroup
		for i := 0; i < rowsA; i++ {
			wg.Add(1)
			go func(rowIdx int) {
				defer wg.Done()
				for j := 0; j < colsB; j++ {
					sum := 0.0
					for k := 0; k < colsA; k++ {
						sum += a[rowIdx][k] * b[k][j]
					}
					result[rowIdx][j] = sum
				}
			}(i)
		}
		wg.Wait()
	}

	return result, nil
}