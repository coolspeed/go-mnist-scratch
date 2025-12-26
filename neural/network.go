package neural

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/coolspeed/go-mnist-scratch/matrix"
	"github.com/coolspeed/go-mnist-scratch/utils"
)

type Network struct {
	W1 *matrix.Matrix
	B1 *matrix.Matrix
	W2 *matrix.Matrix
	B2 *matrix.Matrix
}

func NewNetwork(inputSize, hiddenSize, outputSize int) *Network {
	rand.Seed(time.Now().UnixNano())

	W1, _ := matrix.NewMatrix(inputSize, hiddenSize)
	B1, _ := matrix.NewMatrix(1, hiddenSize)
	W2, _ := matrix.NewMatrix(hiddenSize, outputSize)
	B2, _ := matrix.NewMatrix(1, outputSize)

	for i := 0; i < inputSize; i++ {
		for j := 0; j < hiddenSize; j++ {
			W1.Data[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(inputSize))
		}
	}

	for i := 0; i < hiddenSize; i++ {
		for j := 0; j < outputSize; j++ {
			W2.Data[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(hiddenSize))
		}
	}

	return &Network{
		W1: W1,
		B1: B1,
		W2: W2,
		B2: B2,
	}
}

func (n *Network) Forward(input *matrix.Matrix) (*matrix.Matrix, *matrix.Matrix, *matrix.Matrix) {
	z1, _ := matrix.DotProduct(input, n.W1)
	z1, _ = matrix.Add(z1, n.B1)
	a1 := matrix.Apply(z1, utils.Sigmoid)

	z2, _ := matrix.DotProduct(a1, n.W2)
	z2, _ = matrix.Add(z2, n.B2)
	a2 := matrix.Apply(z2, func(x float64) float64 { return x })

	softmaxOutput := utils.Softmax(a2.Data[0])

	result, _ := matrix.NewMatrix(1, len(softmaxOutput))
	result.Data[0] = softmaxOutput

	return a1, result, z1
}

func (n *Network) Backward(input *matrix.Matrix, a1 *matrix.Matrix, output *matrix.Matrix, z1 *matrix.Matrix, target *matrix.Matrix, learningRate float64) {
	outputError, _ := matrix.Subtract(target, output)

	dW2, _ := matrix.DotProduct(matrix.Transpose(a1), outputError)

	sumError := make([]float64, len(outputError.Data[0]))
	for j := range outputError.Data[0] {
		for i := range outputError.Data {
			sumError[j] += outputError.Data[i][j]
		}
	}

	dB2, _ := matrix.NewMatrix(1, len(sumError))
	dB2.Data[0] = sumError

	hiddenError, _ := matrix.DotProduct(outputError, matrix.Transpose(n.W2))

	sigDeriv := matrix.Apply(z1, utils.SigmoidDerivative)
	for i := range hiddenError.Data {
		for j := range hiddenError.Data[i] {
			hiddenError.Data[i][j] *= sigDeriv.Data[i][j]
		}
	}

	dW1, _ := matrix.DotProduct(matrix.Transpose(input), hiddenError)

	sumHiddenError := make([]float64, len(hiddenError.Data[0]))
	for j := range hiddenError.Data[0] {
		for i := range hiddenError.Data {
			sumHiddenError[j] += hiddenError.Data[i][j]
		}
	}

	dB1, _ := matrix.NewMatrix(1, len(sumHiddenError))
	dB1.Data[0] = sumHiddenError

	for i := 0; i < dW1.Rows; i++ {
		for j := 0; j < dW1.Cols; j++ {
			n.W1.Data[i][j] += dW1.Data[i][j] * learningRate
		}
	}

	for i := 0; i < dB1.Cols; i++ {
		n.B1.Data[0][i] += dB1.Data[0][i] * learningRate
	}

	for i := 0; i < dW2.Rows; i++ {
		for j := 0; j < dW2.Cols; j++ {
			n.W2.Data[i][j] += dW2.Data[i][j] * learningRate
		}
	}

	for i := 0; i < dB2.Cols; i++ {
		n.B2.Data[0][i] += dB2.Data[0][i] * learningRate
	}
}

func (n *Network) Train(images []float64, labels []float64, epochs int, batchSize int, learningRate float64) {
	input, _ := matrix.NewMatrix(1, len(images))
	input.Data[0] = images

	target, _ := matrix.NewMatrix(1, len(labels))
	target.Data[0] = labels

	for epoch := 0; epoch < epochs; epoch++ {
		a1, output, z1 := n.Forward(input)

		correct := 0
		for i := range output.Data[0] {
			if (output.Data[0][i] > 0.5 && labels[i] == 1.0) || (output.Data[0][i] <= 0.5 && labels[i] == 0.0) {
				correct++
			}
		}
		accuracy := float64(correct) / float64(len(labels))

		n.Backward(input, a1, output, z1, target, learningRate)

		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Accuracy: %.2f%%\n", epoch, accuracy*100)
		}
	}
}

func (n *Network) Predict(images []float64) int {
	input, _ := matrix.NewMatrix(1, len(images))
	input.Data[0] = images

	_, output, _ := n.Forward(input)

	maxIdx := 0
	maxVal := output.Data[0][0]
	for i := 1; i < len(output.Data[0]); i++ {
		if output.Data[0][i] > maxVal {
			maxVal = output.Data[0][i]
			maxIdx = i
		}
	}

	return maxIdx
}

func (n *Network) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", n.W1.Rows, n.W1.Cols)
	for i := 0; i < n.W1.Rows; i++ {
		for j := 0; j < n.W1.Cols; j++ {
			fmt.Fprintf(file, "%f ", n.W1.Data[i][j])
		}
	}

	fmt.Fprintf(file, "\n%d\n", n.B1.Cols)
	for j := 0; j < n.B1.Cols; j++ {
		fmt.Fprintf(file, "%f ", n.B1.Data[0][j])
	}

	fmt.Fprintf(file, "\n%d %d\n", n.W2.Rows, n.W2.Cols)
	for i := 0; i < n.W2.Rows; i++ {
		for j := 0; j < n.W2.Cols; j++ {
			fmt.Fprintf(file, "%f ", n.W2.Data[i][j])
		}
	}

	fmt.Fprintf(file, "\n%d\n", n.B2.Cols)
	for j := 0; j < n.B2.Cols; j++ {
		fmt.Fprintf(file, "%f ", n.B2.Data[0][j])
	}

	return nil
}

func (n *Network) LoadModel(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	var rows, cols int
	fmt.Fscanf(file, "%d %d", &rows, &cols)

	W1, _ := matrix.NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Fscanf(file, "%f", &W1.Data[i][j])
		}
	}

	fmt.Fscanf(file, "%d", &cols)
	B1, _ := matrix.NewMatrix(1, cols)
	for j := 0; j < cols; j++ {
		fmt.Fscanf(file, "%f", &B1.Data[0][j])
	}

	fmt.Fscanf(file, "%d %d", &rows, &cols)
	W2, _ := matrix.NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			fmt.Fscanf(file, "%f", &W2.Data[i][j])
		}
	}

	fmt.Fscanf(file, "%d", &cols)
	B2, _ := matrix.NewMatrix(1, cols)
	for j := 0; j < cols; j++ {
		fmt.Fscanf(file, "%f", &B2.Data[0][j])
	}

	n.W1 = W1
	n.B1 = B1
	n.W2 = W2
	n.B2 = B2

	return nil
}
