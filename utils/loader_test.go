package utils

import (
	"path/filepath"
	"testing"
)

// Define constants for the expected MNIST dataset properties
const (
	expectedTrainImages = 60000
	expectedTestImages  = 10000
	expectedImageRows   = 28
	expectedImageCols   = 28
	numClasses          = 10
)

func TestLoadImages(t *testing.T) {
	// Assuming setup.sh has been run and data files are in ./data
	trainImagePath := filepath.Join("..", "data", "train-images-idx3-ubyte.gz")
	
	imageData, err := LoadImages(trainImagePath)
	if err != nil {
		t.Fatalf("Failed to load training images: %v", err)
	}

	if imageData.NumImages != expectedTrainImages {
		t.Errorf("Expected %d training images, got %d", expectedTrainImages, imageData.NumImages)
	}
	if imageData.NumRows != expectedImageRows {
		t.Errorf("Expected %d image rows, got %d", expectedImageRows, imageData.NumRows)
	}
	if imageData.NumCols != expectedImageCols {
		t.Errorf("Expected %d image columns, got %d", expectedImageCols, imageData.NumCols)
	}
	if len(imageData.Images) != int(imageData.NumImages) {
		t.Errorf("Images slice length mismatch. Expected %d, got %d", imageData.NumImages, len(imageData.Images))
	}
	if len(imageData.Images[0]) != int(imageData.NumRows*imageData.NumCols) {
		t.Errorf("First image pixel count mismatch. Expected %d, got %d", imageData.NumRows*imageData.NumCols, len(imageData.Images[0]))
	}

	// Check normalization: pixel values should be between 0.0 and 1.0
	for _, img := range imageData.Images {
		for _, pixel := range img {
			if pixel < 0.0 || pixel > 1.0 {
				t.Errorf("Pixel value %f out of normalized range [0.0, 1.0]", pixel)
			}
		}
	}

	testImagePath := filepath.Join("..", "data", "t10k-images-idx3-ubyte.gz")
	testImageData, err := LoadImages(testImagePath)
	if err != nil {
		t.Fatalf("Failed to load test images: %v", err)
	}
	if testImageData.NumImages != expectedTestImages {
		t.Errorf("Expected %d test images, got %d", expectedTestImages, testImageData.NumImages)
	}
}

func TestLoadLabels(t *testing.T) {
	trainLabelPath := filepath.Join("..", "data", "train-labels-idx1-ubyte.gz")
	labelData, err := LoadLabels(trainLabelPath)
	if err != nil {
		t.Fatalf("Failed to load training labels: %v", err)
	}

	if labelData.NumLabels != expectedTrainImages {
		t.Errorf("Expected %d training labels, got %d", expectedTrainImages, labelData.NumLabels)
	}
	if len(labelData.Labels) != int(labelData.NumLabels) {
		t.Errorf("Labels slice length mismatch. Expected %d, got %d", labelData.NumLabels, len(labelData.Labels))
	}

	// Check if labels are within valid range (0-9)
	for _, label := range labelData.Labels {
		if label < 0 || label > 9 {
			t.Errorf("Label value %d out of valid range [0, 9]", label)
		}
	}

	testLabelPath := filepath.Join("..", "data", "t10k-labels-idx1-ubyte.gz")
	testLabelData, err := LoadLabels(testLabelPath)
	if err != nil {
		t.Fatalf("Failed to load test labels: %v", err)
	}
	if testLabelData.NumLabels != expectedTestImages {
		t.Errorf("Expected %d test labels, got %d", expectedTestImages, testLabelData.NumLabels)
	}
}

func TestOneHotEncode(t *testing.T) {
	tests := []struct {
		label    uint8
		expected []float64
	}{
		{0, []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
		{3, []float64{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
		{9, []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}},
		// Test case for out-of-bounds label (should still return a 10-element slice of zeros)
		{10, []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
	}

	for _, tt := range tests {
		result := OneHotEncode(tt.label, numClasses)
		if len(result) != numClasses {
			t.Errorf("OneHotEncode(%d) result length mismatch. Expected %d, got %d", tt.label, numClasses, len(result))
			continue
		}
		for i, val := range result {
			if val != tt.expected[i] {
				t.Errorf("OneHotEncode(%d) at index %d mismatch. Expected %f, got %f", tt.label, i, tt.expected[i], val)
			}
		}
	}
}

func TestLoadMNIST(t *testing.T) {
	trainImagePath := filepath.Join("..", "data", "train-images-idx3-ubyte.gz")
	trainLabelPath := filepath.Join("..", "data", "train-labels-idx1-ubyte.gz")

	imageDataSet, labelDataSet, err := LoadMNIST(trainImagePath, trainLabelPath)
	if err != nil {
		t.Fatalf("Failed to load MNIST dataset: %v", err)
	}

	if imageDataSet == nil || labelDataSet == nil {
		t.Fatalf("LoadMNIST returned nil datasets")
	}

	if imageDataSet.NumImages != expectedTrainImages {
		t.Errorf("Expected %d images, got %d", expectedTrainImages, imageDataSet.NumImages)
	}

	// Further checks would be redundant as LoadImages and LoadLabels are tested separately.
	// This primarily checks the orchestration of calling both loaders.
}
