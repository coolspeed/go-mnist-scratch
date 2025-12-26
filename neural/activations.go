package neural

import "math"

// Sigmoid function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidPrime calculates the derivative of the sigmoid function
func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// Softmax applies the Softmax function to a slice of float64 values.
// It normalizes the values into a probability distribution.
func Softmax(inputs []float64) []float64 {
	expSum := 0.0
	outputs := make([]float64, len(inputs))

	// Calculate exponentials and their sum
	for _, input := range inputs {
		expVal := math.Exp(input)
		expSum += expVal
		// Store expVal temporarily for later division.
		// Note: This approach re-calculates math.Exp, which is slightly inefficient
		// but avoids storing a separate slice for intermediate exp values.
		// For performance-critical applications, consider storing exp(input)
		// in a separate slice before calculating sum and then outputs.
		// However, for this project's scope, this is acceptable.
	}

	// Calculate softmax probabilities
	for i, input := range inputs {
		outputs[i] = math.Exp(input) / expSum
	}
	return outputs
}
