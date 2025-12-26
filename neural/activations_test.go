package neural

import (
	"math"
	"testing"
)

const float64EqualityThreshold = 1e-9

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= float64EqualityThreshold
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.5},
		{1.0, 0.73105857863},
		{-1.0, 0.26894142137},
		{10.0, 0.99995460213},
		{-10.0, 0.00004539786},
	}

	for _, tt := range tests {
		result := Sigmoid(tt.input)
		if !almostEqual(result, tt.expected) {
			t.Errorf("Sigmoid(%f): expected %f, got %f", tt.input, tt.expected, result)
		}
	}
}

func TestSigmoidPrime(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.25},
		{1.0, 0.19661193324}, // sigmoid(1) * (1 - sigmoid(1))
		{-1.0, 0.19661193324}, // sigmoid(-1) * (1 - sigmoid(-1))
		{10.0, 0.00004539566},
		{-10.0, 0.00004539566},
	}

	for _, tt := range tests {
		result := SigmoidPrime(tt.input)
		if !almostEqual(result, tt.expected) {
			t.Errorf("SigmoidPrime(%f): expected %f, got %f", tt.input, tt.expected, result)
		}
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		input    []float64
		expected []float64
	}{
		{[]float64{0.0, 0.0, 0.0}, []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}},
		{[]float64{1.0, 0.0, 0.0}, []float64{0.57611688476, 0.21194155761, 0.21194155761}},
		{[]float64{1.0, 2.0, 3.0}, []float64{0.09003057317, 0.24472847105, 0.66524095577}},
		{[]float64{-1.0, -2.0, -3.0}, []float64{0.66524095577, 0.24472847105, 0.09003057317}},
		// Test with a single element
		{[]float64{0.0}, []float64{1.0}},
		// Test with large values to check for overflow (though math.Exp handles this to some extent)
		{[]float64{100.0, 100.0}, []float64{0.5, 0.5}},
	}

	for _, tt := range tests {
		result := Softmax(tt.input)
		if len(result) != len(tt.expected) {
			t.Errorf("Softmax(%v) result length mismatch: expected %d, got %d", tt.input, len(tt.expected), len(result))
			continue
		}
		for i := range result {
			if !almostEqual(result[i], tt.expected[i]) {
				t.Errorf("Softmax(%v) at index %d: expected %f, got %f", tt.input, i, tt.expected[i], result[i])
			}
		}

		// Check that softmax outputs sum to 1
		sum := 0.0
		for _, val := range result {
			sum += val
		}
		if !almostEqual(sum, 1.0) && len(result) > 0 { // sum should be 1 unless input is empty
			t.Errorf("Softmax(%v) output sum mismatch: expected 1.0, got %f", tt.input, sum)
		}
	}
}
