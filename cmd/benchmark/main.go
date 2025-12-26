package main

import (
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/coolspeed/go-mnist-scratch/matrix"
	"github.com/coolspeed/go-mnist-scratch/neural"
	"github.com/coolspeed/go-mnist-scratch/utils"
)

const (
	inputSize   = 784
	hiddenSize  = 200
	outputSize  = 10
	modelPath   = "mnist_model.gob"
	sampleCount = 10
)

func main() {
	// 1. Load Model
	net := neural.NewNetwork(inputSize, hiddenSize, outputSize, 0.0)
	if err := net.LoadModel(modelPath); err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// 2. Load Test Data
	testImagePath := filepath.Join("data", "t10k-images-idx3-ubyte.gz")
	testLabelPath := filepath.Join("data", "t10k-labels-idx1-ubyte.gz")
	testImagesData, _, err := utils.LoadMNIST(testImagePath, testLabelPath)
	if err != nil {
		log.Fatalf("Error loading data: %v", err)
	}

	fmt.Printf("Benchmarking inference on %d samples...\n", sampleCount)
	fmt.Println("------------------------------------------------")

	var totalDuration time.Duration

	// Prepare matrices in advance to measure only prediction time
	inputs := make([]matrix.Matrix, sampleCount)
	for i := 0; i < sampleCount; i++ {
		inputs[i] = matrix.NewMatrix(1, inputSize)
		inputs[i][0] = testImagesData.Images[i]
	}

	// Warm-up (perform one prediction to load everything into cache/memory)
	_, _ = net.Predict(inputs[0])

	// 3. Measure
	for i := 0; i < sampleCount; i++ {
		start := time.Now()
		
		_, err := net.Predict(inputs[i])
		if err != nil {
			log.Printf("Prediction error: %v", err)
			continue
		}
		
		duration := time.Since(start)
		totalDuration += duration
		
		fmt.Printf("Sample %2d: %v\n", i+1, duration)
	}

	avgDuration := totalDuration / time.Duration(sampleCount)
	fmt.Println("------------------------------------------------")
	fmt.Printf("Total Time: %v\n", totalDuration)
	fmt.Printf("Average Inference Time: %v\n", avgDuration)
	
	if avgDuration < 10*time.Millisecond {
		fmt.Println("✅ Goal Achieved: Under 10ms per inference")
	} else {
		fmt.Println("⚠️ Goal Missed: Over 10ms per inference")
	}
}
