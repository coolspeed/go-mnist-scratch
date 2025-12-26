package main

import (
	"fmt"
	"os"

	"github.com/coolspeed/go-mnist-scratch/neural"
	"github.com/coolspeed/go-mnist-scratch/utils"
)

const (
	inputSize    = 784
	hiddenSize   = 200
	outputSize   = 10
	learningRate = 0.3
	epochs       = 1
	batchSize    = 600
)

func train(imageFile, labelFile string) {
	fmt.Println("Loading MNIST data...")
	data, err := utils.LoadMNIST(imageFile, labelFile)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded %d images and %d labels\n", len(data.Images), len(data.Labels))

	network := neural.NewNetwork(inputSize, hiddenSize, outputSize)

	fmt.Println("Starting training...")

	numBatches := 1
	for epoch := 0; epoch < epochs; epoch++ {
		correct := 0

		for batchIdx := 0; batchIdx < numBatches; batchIdx++ {
			start := batchIdx * batchSize
			end := start + batchSize
			if end > len(data.Images) {
				end = len(data.Images)
			}

			for i := start; i < end; i++ {
				images := utils.FlattenImage(data.Images[i])
				network.Train(images, data.Labels[i].OneHot, 1, 1, learningRate)

				output := network.Predict(images)
				if output == data.Labels[i].Value {
					correct++
				}
			}
		}

		accuracy := float64(correct) / float64(len(data.Images)) * 100
		fmt.Printf("Epoch %d, Accuracy: %.2f%%\n", epoch, accuracy)
	}

	fmt.Println("Training complete!")
	fmt.Println("Saving model...")
	err = network.SaveModel("model.txt")
	if err != nil {
		fmt.Printf("Error saving model: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Model saved to model.txt")
}

func eval() {
	fmt.Println("Loading test data...")
	testData, err := utils.LoadMNIST("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
	if err != nil {
		fmt.Printf("Error loading test data: %v\n", err)
		return
	}

	fmt.Printf("Loaded %d test images\n", len(testData.Images))

	network := &neural.Network{}
	err = network.LoadModel("model.txt")
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}

	correct := 0
	for i := 0; i < len(testData.Images); i++ {
		images := utils.FlattenImage(testData.Images[i])
		prediction := network.Predict(images)
		if prediction == testData.Labels[i].Value {
			correct++
		}

		if (i+1)%1000 == 0 {
			fmt.Printf("Processed %d/%d images\n", i+1, len(testData.Images))
		}
	}

	accuracy := float64(correct) / float64(len(testData.Images)) * 100
	fmt.Printf("\nTest Accuracy: %.2f%% (%d/%d)\n", accuracy, correct, len(testData.Images))
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <train|eval> [args...]")
		fmt.Println("  train <train-images-file> <train-labels-file>  - Train the model")
		fmt.Println("  eval  - Evaluate the model on test data")
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "train":
		if len(os.Args) < 4 {
			fmt.Println("Usage: go run main.go train <train-images-file> <train-labels-file>")
			os.Exit(1)
		}
		train(os.Args[2], os.Args[3])
	case "eval":
		eval()
	default:
		fmt.Printf("Unknown command: %s\n", command)
		fmt.Println("Available commands: train, eval")
		os.Exit(1)
	}
}
