package main

import (
	"fmt"
	"log"
	"math/rand"
	"path/filepath"
	"time"

	"github.com/coolspeed/go-mnist-scratch/matrix"
	"github.com/coolspeed/go-mnist-scratch/neural"
	"github.com/coolspeed/go-mnist-scratch/utils"
)

const (
	inputSize   = 784 // 28x28 pixels
	hiddenSize  = 200
	outputSize  = 10 // 0-9 digits
	learningRate = 0.3
	epochs      = 20
	batchSize   = 64
	modelPath   = "mnist_model.gob" // Using .gob for now, CLAUDE.md suggests JSON/Gob
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("Loading MNIST data...")
	trainImagePath := filepath.Join("data", "train-images-idx3-ubyte.gz")
	trainLabelPath := filepath.Join("data", "train-labels-idx1-ubyte.gz")
	testImagePath := filepath.Join("data", "t10k-images-idx3-ubyte.gz")
	testLabelPath := filepath.Join("data", "t10k-labels-idx1-ubyte.gz")

	trainImagesData, trainLabelsData, err := utils.LoadMNIST(trainImagePath, trainLabelPath)
	if err != nil {
		log.Fatalf("Error loading training data: %v", err)
	}
	testImagesData, testLabelsData, err := utils.LoadMNIST(testImagePath, testLabelPath)
	if err != nil {
		log.Fatalf("Error loading testing data: %v", err)
	}

	fmt.Printf("Training images loaded: %d\n", trainImagesData.NumImages)
	fmt.Printf("Training labels loaded: %d\n", trainLabelsData.NumLabels)
	fmt.Printf("Test images loaded: %d\n", testImagesData.NumImages)
	fmt.Printf("Test labels loaded: %d\n", testLabelsData.NumLabels)

	net := neural.NewNetwork(inputSize, hiddenSize, outputSize, learningRate)

	fmt.Println("Starting training...")
	for e := 0; e < epochs; e++ {
		fmt.Printf("Epoch %d/%d\n", e+1, epochs)

		// Shuffle training data
		perm := rand.Perm(int(trainImagesData.NumImages))

		for i := 0; i < int(trainImagesData.NumImages); i += batchSize {
			end := i + batchSize
			if end > int(trainImagesData.NumImages) {
				end = int(trainImagesData.NumImages)
			}

			// For simplicity, current Train method handles single sample.
			// Batching logic would involve averaging gradients.
			// For now, process each sample in the batch individually.
			for j := i; j < end; j++ {
				idx := perm[j]
				inputMatrix := matrix.NewMatrix(1, inputSize)
				inputMatrix[0] = trainImagesData.Images[idx]

				targetMatrix := matrix.NewMatrix(1, outputSize)
				targetMatrix[0] = utils.OneHotEncode(trainLabelsData.Labels[idx], outputSize)

				err := net.Train(inputMatrix, targetMatrix)
				if err != nil {
					log.Printf("Error training on sample %d: %v", idx, err)
				}
			}
		}

		// Evaluate accuracy on test set after each epoch (optional, but good for monitoring)
		correct := 0
		for i := 0; i < int(testImagesData.NumImages); i++ {
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
		}
		accuracy := float64(correct) / float64(testImagesData.NumImages) * 100
		fmt.Printf("Epoch %d: Test Accuracy: %.2f%%\n", e+1, accuracy)
	}

	fmt.Println("Training complete. Saving model...")
	err = net.SaveModel(modelPath)
	if err != nil {
		log.Fatalf("Error saving model: %v", err)
	}
	fmt.Printf("Model saved to %s\n", modelPath)
}