package matrix

import (
	"errors"
)

type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

func NewMatrix(rows, cols int) (*Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, errors.New("matrix dimensions must be positive")
	}

	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}

	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}, nil
}

func Add(m1, m2 *Matrix) (*Matrix, error) {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return nil, errors.New("matrices must have the same dimensions")
	}

	result, _ := NewMatrix(m1.Rows, m1.Cols)
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			result.Data[i][j] = m1.Data[i][j] + m2.Data[i][j]
		}
	}

	return result, nil
}

func Subtract(m1, m2 *Matrix) (*Matrix, error) {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return nil, errors.New("matrices must have the same dimensions")
	}

	result, _ := NewMatrix(m1.Rows, m1.Cols)
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m1.Cols; j++ {
			result.Data[i][j] = m1.Data[i][j] - m2.Data[i][j]
		}
	}

	return result, nil
}

func Apply(m *Matrix, fn func(float64) float64) *Matrix {
	result, _ := NewMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

func Transpose(m *Matrix) *Matrix {
	result, _ := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func DotProduct(m1, m2 *Matrix) (*Matrix, error) {
	if m1.Cols != m2.Rows {
		return nil, errors.New("incompatible dimensions: m1 columns must equal m2 rows")
	}

	result, _ := NewMatrix(m1.Rows, m2.Cols)
	for i := 0; i < m1.Rows; i++ {
		for j := 0; j < m2.Cols; j++ {
			sum := 0.0
			for k := 0; k < m1.Cols; k++ {
				sum += m1.Data[i][k] * m2.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}

	return result, nil
}
