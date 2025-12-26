package neural

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/coolspeed/go-mnist-scratch/matrix"
)

// Network represents a neural network
type Network struct {
	// Weights and biases for hidden layer
	W1 matrix.Matrix
	B1 matrix.Matrix
	// Weights and biases for output layer
	W2 matrix.Matrix
	B2 matrix.Matrix

	// Learning rate
	LearningRate float64
}

// NewNetwork creates and initializes a new neural network
func NewNetwork(inputSize, hiddenSize, outputSize int, learningRate float64) *Network {
	rand.Seed(time.Now().UnixNano())

	net := &Network{
		LearningRate: learningRate,
	}

	// Initialize weights and biases
	// W1: inputSize x hiddenSize
	net.W1 = matrix.NewMatrix(inputSize, hiddenSize)
	// B1: 1 x hiddenSize
	net.B1 = matrix.NewMatrix(1, hiddenSize)
	// W2: hiddenSize x outputSize
	net.W2 = matrix.NewMatrix(hiddenSize, outputSize)
	// B2: 1 x outputSize
	net.B2 = matrix.NewMatrix(1, outputSize)

	// He initialization for weights (suitable for ReLU, often used with others too)
	// For sigmoid, Xavier might be more appropriate, but given the prompt, this is a reasonable start.
	stdDev1 := math.Sqrt(2.0 / float64(inputSize))
	for i := 0; i < inputSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			net.W1[i][j] = rand.NormFloat64() * stdDev1
		}
	}
	// Biases initialized to zero
	// for i := 0; i < hiddenSize; i++ {
	// 	net.B1[0][i] = 0.0
	// }

	stdDev2 := math.Sqrt(2.0 / float64(hiddenSize))
	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < outputSize; j++ {
			net.W2[i][j] = rand.NormFloat64() * stdDev2
		}
	}
	// Biases initialized to zero
	// for i := 0; i < outputSize; i++ {
	// 	net.B2[0][i] = 0.0
	// }

	return net
}

// Forward performs the forward pass through the network
// Returns:
//   - a1: Activated output of the hidden layer
//   - a2: Activated output of the output layer (Softmax probabilities)
//   - z1: Weighted sum + bias of hidden layer (before activation)
//   - z2: Weighted sum + bias of output layer (before activation)
func (net *Network) Forward(input matrix.Matrix) (a1, a2, z1, z2 matrix.Matrix, err error) {
	// Layer 1 (Hidden Layer)
	z1, err = input.DotProduct(net.W1)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("forward pass error (input.DotProduct(W1)): %w", err)
	}
	z1, err = z1.Add(net.B1)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("forward pass error (z1.Add(B1)): %w", err)
	}
	a1 = z1.Apply(Sigmoid) // Apply Sigmoid activation

	// Layer 2 (Output Layer)
	z2, err = a1.DotProduct(net.W2)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("forward pass error (a1.DotProduct(W2)): %w", err)
	}
	z2, err = z2.Add(net.B2)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("forward pass error (z2.Add(B2)): %w", err)
	}
	// Apply Softmax activation. Softmax operates on a 1D slice of inputs.
	// Assuming z2 is a 1xN matrix, we need to extract its first row.
	if len(z2) == 0 || len(z2[0]) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("z2 matrix is empty, cannot apply Softmax")
	}
	softmaxInputs := z2[0]
	softmaxOutputs := Softmax(softmaxInputs)

	// Convert softmaxOutputs back to a 1xN matrix
	a2 = matrix.NewMatrix(1, len(softmaxOutputs))
	a2[0] = softmaxOutputs

	return a1, a2, z1, z2, nil
}

// Predict performs a forward pass and returns the predicted digit (0-9)
func (net *Network) Predict(input matrix.Matrix) (int, error) {
	_, a2, _, _, err := net.Forward(input)
	if err != nil {
		return -1, fmt.Errorf("prediction error: %w", err)
	}

	maxVal := -1.0
	prediction := -1

	// Assuming a2 is a 1xN matrix of probabilities
	if len(a2) == 0 || len(a2[0]) == 0 {
		return -1, fmt.Errorf("output layer is empty")
	}

	for i, val := range a2[0] {
		if val > maxVal {
			maxVal = val
			prediction = i
		}
	}
	return prediction, nil
}




// Train trains the neural network using backpropagation
// This is a placeholder and will need full implementation for backpropagation.
func (net *Network) Train(input, target matrix.Matrix) error {
	// Forward pass
	a1, output, z1, _, err := net.Forward(input)
	if err != nil {
		return fmt.Errorf("training forward pass failed: %w", err)
	}

	// Backpropagation
	// Output Layer Error (delta2)
	// For Softmax with Cross-Entropy Loss, delta2 = output - target
	delta2, err := output.Subtract(target)
	if err != nil {
		return fmt.Errorf("training output error calculation failed: %w", err)
	}

	// Update W2 and B2
	// dW2 = a1_T . delta2
	a1T := a1.Transpose()
	dW2, err := a1T.DotProduct(delta2)
	if err != nil {
		return fmt.Errorf("training dW2 calculation failed: %w", err)
	}
	
	// dB2 is simply sum of delta2 rows, assuming delta2 is 1xN
	// If batch training, sum of deltas for the batch
	// For a single sample (1xN matrix), dB2 = delta2
	dB2 := delta2

	// Hidden Layer Error (delta1)
	// delta1 = (delta2 . W2_T) * sigmoid_prime(z1)
	w2T := net.W2.Transpose()
	delta1Intermediate, err := delta2.DotProduct(w2T)
	if err != nil {
		return fmt.Errorf("training delta1 intermediate calculation failed: %w", err)
	}
	
z1SigmoidPrime := z1.Apply(SigmoidPrime) // Derivative of sigmoid applied to z1
	
	delta1, err := delta1Intermediate.MultiplyElementWise(z1SigmoidPrime) // Custom element-wise multiplication
	if err != nil {
		return fmt.Errorf("training delta1 calculation failed: %w", err)
	}

	// Update W1 and B1
	// dW1 = input_T . delta1
	inputT := input.Transpose()
	dW1, err := inputT.DotProduct(delta1)
	if err != nil {
		return fmt.Errorf("training dW1 calculation failed: %w", err)
	}
	// dB1 = delta1 (for single sample)
	dB1 := delta1

	// Apply gradient descent updates
	// W2 = W2 - LearningRate * dW2
	scaled_dW2 := dW2.ScalarMultiply(net.LearningRate) // Implement ScalarMultiply
	net.W2, err = net.W2.Subtract(scaled_dW2)
	if err != nil {
		return fmt.Errorf("training W2 update failed: %w", err)
	}

	scaled_dB2 := dB2.ScalarMultiply(net.LearningRate)
	net.B2, err = net.B2.Subtract(scaled_dB2)
	if err != nil {
		return fmt.Errorf("training B2 update failed: %w", err)
	}

	// W1 = W1 - LearningRate * dW1
	scaled_dW1 := dW1.ScalarMultiply(net.LearningRate)
	net.W1, err = net.W1.Subtract(scaled_dW1)
	if err != nil {
		return fmt.Errorf("training W1 update failed: %w", err)
	}

	scaled_dB1 := dB1.ScalarMultiply(net.LearningRate)
	net.B1, err = net.B1.Subtract(scaled_dB1)
	if err != nil {
		return fmt.Errorf("training B1 update failed: %w", err)
	}

	return nil
}




// SaveModel saves the network's weights and biases to a file.
// This is a simple text-based save. For production, consider JSON/Gob.
func (net *Network) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Helper to write a matrix
	writeMatrix := func(m matrix.Matrix) error {
		_, err := fmt.Fprintf(file, "%d %d\n", len(m), len(m[0]))
		if err != nil {
			return err
		}
		for i := range m {
			for j := range m[i] {
				_, err = fmt.Fprintf(file, "%f ", m[i][j])
				if err != nil {
					return err
				}
			}
			_, err = fmt.Fprintln(file) // Newline after each row
			if err != nil {
				return err
			}
		}
		return nil
	}

	fmt.Fprintln(file, "W1")
	if err := writeMatrix(net.W1); err != nil {
		return fmt.Errorf("failed to write W1: %w", err)
	}
	fmt.Fprintln(file, "B1")
	if err := writeMatrix(net.B1); err != nil {
		return fmt.Errorf("failed to write B1: %w", err)
	}
	fmt.Fprintln(file, "W2")
	if err := writeMatrix(net.W2); err != nil {
		return fmt.Errorf("failed to write W2: %w", err)
	}
	fmt.Fprintln(file, "B2")
	if err := writeMatrix(net.B2); err != nil {
		return fmt.Errorf("failed to write B2: %w", err)
	}

	return nil
}

// LoadModel loads the network's weights and biases from a file.
func (net *Network) LoadModel(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Helper to read a matrix
	readMatrix := func() (matrix.Matrix, error) {
		var rows, cols int
		_, err := fmt.Fscanf(file, "%d %d\n", &rows, &cols)
		if err != nil {
			return nil, err
		}
		m := matrix.NewMatrix(rows, cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				_, err = fmt.Fscanf(file, "%f", &m[i][j])
				if err != nil {
					return nil, err
				}
			}
		}
		return m, nil
	}

	var header string

	fmt.Fscanln(file, &header) // Read "W1"
	net.W1, err = readMatrix()
	if err != nil {
		return fmt.Errorf("failed to read W1: %w", err)
	}
	fmt.Fscanln(file, &header) // Read "B1"
	net.B1, err = readMatrix()
	if err != nil {
		return fmt.Errorf("failed to read B1: %w", err)
	}
	fmt.Fscanln(file, &header) // Read "W2"
	net.W2, err = readMatrix()
	if err != nil {
		return fmt.Errorf("failed to read W2: %w", err)
	}
	fmt.Fscanln(file, &header) // Read "B2"
	net.B2, err = readMatrix()
	if err != nil {
		return fmt.Errorf("failed to read B2: %w", err)
	}

	return nil
}
