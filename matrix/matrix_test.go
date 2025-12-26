package matrix

import (
	"testing"
)

func TestNewMatrix(t *testing.T) {
	tests := []struct {
		name    string
		rows    int
		cols    int
		wantErr bool
	}{
		{
			name:    "valid 2x3 matrix",
			rows:    2,
			cols:    3,
			wantErr: false,
		},
		{
			name:    "zero rows",
			rows:    0,
			cols:    3,
			wantErr: true,
		},
		{
			name:    "zero cols",
			rows:    2,
			cols:    0,
			wantErr: true,
		},
		{
			name:    "negative rows",
			rows:    -1,
			cols:    3,
			wantErr: true,
		},
		{
			name:    "negative cols",
			rows:    2,
			cols:    -1,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewMatrix(tt.rows, tt.cols)
			if tt.wantErr {
				if err == nil {
					t.Errorf("NewMatrix() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("NewMatrix() unexpected error: %v", err)
				return
			}
			if m.Rows != tt.rows {
				t.Errorf("NewMatrix() Rows = %v, want %v", m.Rows, tt.rows)
			}
			if m.Cols != tt.cols {
				t.Errorf("NewMatrix() Cols = %v, want %v", m.Cols, tt.cols)
			}
			if len(m.Data) != tt.rows {
				t.Errorf("NewMatrix() Data length = %v, want %v", len(m.Data), tt.rows)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	m1, _ := NewMatrix(2, 2)
	m1.Data[0][0] = 1
	m1.Data[0][1] = 2
	m1.Data[1][0] = 3
	m1.Data[1][1] = 4

	m2, _ := NewMatrix(2, 2)
	m2.Data[0][0] = 5
	m2.Data[0][1] = 6
	m2.Data[1][0] = 7
	m2.Data[1][1] = 8

	tests := []struct {
		name    string
		m1      *Matrix
		m2      *Matrix
		wantErr bool
		want    [][]float64
	}{
		{
			name:    "valid addition",
			m1:      m1,
			m2:      m2,
			wantErr: false,
			want:    [][]float64{{6, 8}, {10, 12}},
		},
		{
			name:    "different dimensions",
			m1:      m1,
			m2:      func() *Matrix { m, _ := NewMatrix(2, 3); return m }(),
			wantErr: true,
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Add(tt.m1, tt.m2)
			if tt.wantErr {
				if err == nil {
					t.Errorf("Add() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("Add() unexpected error: %v", err)
				return
			}
			for i := range tt.want {
				for j := range tt.want[i] {
					if result.Data[i][j] != tt.want[i][j] {
						t.Errorf("Add() Data[%d][%d] = %v, want %v", i, j, result.Data[i][j], tt.want[i][j])
					}
				}
			}
		})
	}
}

func TestSubtract(t *testing.T) {
	m1, _ := NewMatrix(2, 2)
	m1.Data[0][0] = 5
	m1.Data[0][1] = 6
	m1.Data[1][0] = 7
	m1.Data[1][1] = 8

	m2, _ := NewMatrix(2, 2)
	m2.Data[0][0] = 1
	m2.Data[0][1] = 2
	m2.Data[1][0] = 3
	m2.Data[1][1] = 4

	tests := []struct {
		name    string
		m1      *Matrix
		m2      *Matrix
		wantErr bool
		want    [][]float64
	}{
		{
			name:    "valid subtraction",
			m1:      m1,
			m2:      m2,
			wantErr: false,
			want:    [][]float64{{4, 4}, {4, 4}},
		},
		{
			name:    "different dimensions",
			m1:      m1,
			m2:      func() *Matrix { m, _ := NewMatrix(2, 3); return m }(),
			wantErr: true,
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Subtract(tt.m1, tt.m2)
			if tt.wantErr {
				if err == nil {
					t.Errorf("Subtract() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("Subtract() unexpected error: %v", err)
				return
			}
			for i := range tt.want {
				for j := range tt.want[i] {
					if result.Data[i][j] != tt.want[i][j] {
						t.Errorf("Subtract() Data[%d][%d] = %v, want %v", i, j, result.Data[i][j], tt.want[i][j])
					}
				}
			}
		})
	}
}

func TestApply(t *testing.T) {
	m, _ := NewMatrix(2, 2)
	m.Data[0][0] = 1
	m.Data[0][1] = 2
	m.Data[1][0] = 3
	m.Data[1][1] = 4

	fn := func(x float64) float64 { return x * 2 }
	result := Apply(m, fn)
	want := [][]float64{{2, 4}, {6, 8}}

	for i := range want {
		for j := range want[i] {
			if result.Data[i][j] != want[i][j] {
				t.Errorf("Apply() Data[%d][%d] = %v, want %v", i, j, result.Data[i][j], want[i][j])
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	m, _ := NewMatrix(2, 3)
	m.Data[0][0] = 1
	m.Data[0][1] = 2
	m.Data[0][2] = 3
	m.Data[1][0] = 4
	m.Data[1][1] = 5
	m.Data[1][2] = 6

	result := Transpose(m)

	if result.Rows != 3 {
		t.Errorf("Transpose() Rows = %v, want 3", result.Rows)
	}
	if result.Cols != 2 {
		t.Errorf("Transpose() Cols = %v, want 2", result.Cols)
	}

	want := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	for i := range want {
		for j := range want[i] {
			if result.Data[i][j] != want[i][j] {
				t.Errorf("Transpose() Data[%d][%d] = %v, want %v", i, j, result.Data[i][j], want[i][j])
			}
		}
	}
}

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name    string
		m1      *Matrix
		m2      *Matrix
		wantErr bool
		want    [][]float64
	}{
		{
			name: "2x3 times 3x2",
			m1: func() *Matrix {
				m, _ := NewMatrix(2, 3)
				m.Data[0][0] = 1
				m.Data[0][1] = 2
				m.Data[0][2] = 3
				m.Data[1][0] = 4
				m.Data[1][1] = 5
				m.Data[1][2] = 6
				return m
			}(),
			m2: func() *Matrix {
				m, _ := NewMatrix(3, 2)
				m.Data[0][0] = 7
				m.Data[0][1] = 8
				m.Data[1][0] = 9
				m.Data[1][1] = 10
				m.Data[2][0] = 11
				m.Data[2][1] = 12
				return m
			}(),
			wantErr: false,
			want:    [][]float64{{58, 64}, {139, 154}},
		},
		{
			name: "incompatible dimensions",
			m1: func() *Matrix {
				m, _ := NewMatrix(2, 3)
				return m
			}(),
			m2: func() *Matrix {
				m, _ := NewMatrix(2, 2)
				return m
			}(),
			wantErr: true,
			want:    nil,
		},
		{
			name: "1x1 times 1x1",
			m1: func() *Matrix {
				m, _ := NewMatrix(1, 1)
				m.Data[0][0] = 3
				return m
			}(),
			m2: func() *Matrix {
				m, _ := NewMatrix(1, 1)
				m.Data[0][0] = 4
				return m
			}(),
			wantErr: false,
			want:    [][]float64{{12}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := DotProduct(tt.m1, tt.m2)
			if tt.wantErr {
				if err == nil {
					t.Errorf("DotProduct() expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("DotProduct() unexpected error: %v", err)
				return
			}
			for i := range tt.want {
				for j := range tt.want[i] {
					if result.Data[i][j] != tt.want[i][j] {
						t.Errorf("DotProduct() Data[%d][%d] = %v, want %v", i, j, result.Data[i][j], tt.want[i][j])
					}
				}
			}
		})
	}
}
