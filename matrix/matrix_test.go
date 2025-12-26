package matrix

import (
	"math"
	"testing"
)

func TestNewMatrix(t *testing.T) {
	rows, cols := 3, 4
	m := NewMatrix(rows, cols)

	if len(m) != rows {
		t.Errorf("Expected %d rows, got %d", rows, len(m))
	}
	for i := 0; i < rows; i++ {
		if len(m[i]) != cols {
			t.Errorf("Row %d: Expected %d columns, got %d", i, cols, len(m[i]))
		}
		for j := 0; j < cols; j++ {
			if m[i][j] != 0.0 {
				t.Errorf("New matrix element at (%d, %d) should be 0.0, got %f", i, j, m[i][j])
			}
		}
	}
}

func TestDotProduct(t *testing.T) {
	// Test case 1: Valid multiplication
	a := Matrix{{1, 2}, {3, 4}}
	b := Matrix{{5, 6}, {7, 8}}
	expected := Matrix{{19, 22}, {43, 50}}
	result, err := a.DotProduct(b)

	if err != nil {
		t.Errorf("DotProduct returned an unexpected error: %v", err)
	}
	if !equalMatrices(result, expected) {
		t.Errorf("DotProduct result mismatch.\nExpected: %v\nGot: %v", expected, result)
	}

	// Test case 2: Incompatible dimensions
	c := Matrix{{1, 2, 3}}
	d := Matrix{{4}, {5}} // This should be 2x1 to be incompatible with 1x3, so it's 1x3 and 2x1 -> error
	_, err = c.DotProduct(d)
	if err == nil {
		t.Error("DotProduct should return an error for incompatible dimensions, but didn't")
	}
	
	// Test case 3: 1x1 matrices
	e := Matrix{{2}}
	f := Matrix{{3}}
	expectedEF := Matrix{{6}}
	resultEF, err := e.DotProduct(f)
	if err != nil {
		t.Errorf("DotProduct for 1x1 matrices returned an unexpected error: %v", err)
	}
	if !equalMatrices(resultEF, expectedEF) {
		t.Errorf("DotProduct 1x1 result mismatch.\nExpected: %v\nGot: %v", expectedEF, resultEF)
	}

	// Test case 4: Matrix * Vector (represented as a matrix)
	g := Matrix{{1, 2, 3}, {4, 5, 6}} // 2x3
	h := Matrix{{7}, {8}, {9}}       // 3x1
	expectedGH := Matrix{{50}, {122}} // 2x1
	resultGH, err := g.DotProduct(h)
	if err != nil {
		t.Errorf("DotProduct for matrix * vector returned an unexpected error: %v", err)
	}
	if !equalMatrices(resultGH, expectedGH) {
		t.Errorf("DotProduct matrix * vector result mismatch.\nExpected: %v\nGot: %v", expectedGH, resultGH)
	}
}

func TestAdd(t *testing.T) {
	// Test case 1: Valid addition
	a := Matrix{{1, 2}, {3, 4}}
	b := Matrix{{5, 6}, {7, 8}}
	expected := Matrix{{6, 8}, {10, 12}}
	result, err := a.Add(b)

	if err != nil {
		t.Errorf("Add returned an unexpected error: %v", err)
	}
	if !equalMatrices(result, expected) {
		t.Errorf("Add result mismatch.\nExpected: %v\nGot: %v", expected, result)
	}

	// Test case 2: Incompatible dimensions
	c := Matrix{{1, 2, 3}}
	d := Matrix{{4, 5}}
	_, err = c.Add(d)
	if err == nil {
		t.Error("Add should return an error for incompatible dimensions, but didn't")
	}
}

func TestSubtract(t *testing.T) {
	// Test case 1: Valid subtraction
	a := Matrix{{5, 6}, {7, 8}}
	b := Matrix{{1, 2}, {3, 4}}
	expected := Matrix{{4, 4}, {4, 4}}
	result, err := a.Subtract(b)

	if err != nil {
		t.Errorf("Subtract returned an unexpected error: %v", err)
	}
	if !equalMatrices(result, expected) {
		t.Errorf("Subtract result mismatch.\nExpected: %v\nGot: %v", expected, result)
	}

	// Test case 2: Incompatible dimensions
	c := Matrix{{1, 2, 3}}
	d := Matrix{{4, 5}}
	_, err = c.Subtract(d)
	if err == nil {
		t.Error("Subtract should return an error for incompatible dimensions, but didn't")
	}
}

func TestApply(t *testing.T) {
	m := Matrix{{1, 2}, {3, 4}}
	fn := func(x float64) float64 {
		return x * x
	}
	expected := Matrix{{1, 4}, {9, 16}}
	result := m.Apply(fn)

	if !equalMatrices(result, expected) {
		t.Errorf("Apply result mismatch.\nExpected: %v\nGot: %v", expected, result)
	}
}

func TestTranspose(t *testing.T) {
	m := Matrix{{1, 2, 3}, {4, 5, 6}}
	expected := Matrix{{1, 4}, {2, 5}, {3, 6}}
	result := m.Transpose()

	if !equalMatrices(result, expected) {
		t.Errorf("Transpose result mismatch.\nExpected: %v\nGot: %v", expected, result)
	}

	// Test case for a square matrix
	squareMatrix := Matrix{{1, 2}, {3, 4}}
	expectedSquare := Matrix{{1, 3}, {2, 4}}
	resultSquare := squareMatrix.Transpose()
	if !equalMatrices(resultSquare, expectedSquare) {
		t.Errorf("Transpose square matrix result mismatch.\nExpected: %v\nGot: %v", expectedSquare, resultSquare)
	}

	// Test case for a 1x1 matrix
	oneByOne := Matrix{{7}}
	expectedOneByOne := Matrix{{7}}
	resultOneByOne := oneByOne.Transpose()
	if !equalMatrices(resultOneByOne, expectedOneByOne) {
		t.Errorf("Transpose 1x1 matrix result mismatch.\nExpected: %v\nGot: %v", expectedOneByOne, resultOneByOne)
	}

	// Test case for empty matrix
	emptyMatrix := NewMatrix(0,0)
	expectedEmpty := NewMatrix(0,0)
	resultEmpty := emptyMatrix.Transpose()
	if !equalMatrices(resultEmpty, expectedEmpty) {
		t.Errorf("Transpose empty matrix result mismatch.\nExpected: %v\nGot: %v", expectedEmpty, resultEmpty)
	}
}

// Helper function to compare two matrices for equality with a small tolerance for float comparisons
func equalMatrices(m1, m2 Matrix) bool {
	if len(m1) != len(m2) {
		return false
	}
	if len(m1) > 0 && len(m1[0]) != len(m2[0]) { // Check column length only if there are rows
		return false
	}
	for i := range m1 {
		if len(m1[i]) != len(m2[i]) {
			return false
		}
		for j := range m1[i] {
			if math.Abs(m1[i][j]-m2[i][j]) > 1e-9 { // Use a small epsilon for float comparison
				return false
			}
		}
	}
	return true
}
