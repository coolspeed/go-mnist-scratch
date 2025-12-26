package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/coolspeed/go-mnist-scratch/matrix"
	"github.com/coolspeed/go-mnist-scratch/neural"
	"github.com/coolspeed/go-mnist-scratch/utils"
)

const (
	inputSize   = 784 // 28x28 pixels
	hiddenSize  = 200
	outputSize  = 10 // 0-9 digits
	modelPath   = "mnist_model.gob"
)

func main() {
	// 1. Load Model
	fmt.Printf("Loading model from %s...\n", modelPath)
	net := neural.NewNetwork(inputSize, hiddenSize, outputSize, 0.0) // Learning rate doesn't matter for inference
	err := net.LoadModel(modelPath)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}
	fmt.Println("Model loaded successfully.")

	// 2. Load Test Data
	fmt.Println("Loading MNIST test data...")
	testImagePath := filepath.Join("data", "t10k-images-idx3-ubyte.gz")
	testLabelPath := filepath.Join("data", "t10k-labels-idx1-ubyte.gz")

	testImagesData, testLabelsData, err := utils.LoadMNIST(testImagePath, testLabelPath)
	if err != nil {
		log.Fatalf("Error loading testing data: %v", err)
	}
	fmt.Printf("Test images loaded: %d\n", testImagesData.NumImages)

	// 3. Evaluate Accuracy
	fmt.Println("Starting evaluation...")
	correct := 0
	total := int(testImagesData.NumImages)

	for i := 0; i < total; i++ {
		inputMatrix := matrix.NewMatrix(1, inputSize)
		inputMatrix[0] = testImagesData.Images[i]

		predicted, err := net.Predict(inputMatrix)
		if err != nil {
			log.Printf("Error predicting for test sample %d: %v", i, err)
			continue
		}

		actual := int(testLabelsData.Labels[i])
		if predicted == actual {
			correct++
		}
		
		if (i+1)%1000 == 0 {
			fmt.Printf("Processed %d/%d samples...\n", i+1, total)
		}
	}

	accuracy := float64(correct) / float64(total) * 100
	fmt.Printf("\nFinal Accuracy: %.2f%%\n", accuracy)
}
