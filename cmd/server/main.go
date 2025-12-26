package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/coolspeed/go-mnist-scratch/matrix"
	"github.com/coolspeed/go-mnist-scratch/neural"
)

const (
	modelPath   = "mnist_model.gob"
	port        = ":8080"
	inputSize   = 784 // 28x28 pixels
	hiddenSize  = 200
	outputSize  = 10 // 0-9 digits
	learningRate = 0.3 // Only used for NewNetwork if no model is loaded.
)

var net *neural.Network

func init() {
	// Initialize network (dummy values for NewNetwork, will be overwritten by LoadModel)
	net = neural.NewNetwork(inputSize, hiddenSize, outputSize, learningRate)

	// Attempt to load the pre-trained model
	fmt.Printf("Loading model from %s...\n", modelPath)
	err := net.LoadModel(modelPath)
	if err != nil {
		fmt.Printf("Warning: Could not load model from %s. Starting with a fresh network. Error: %v\n", modelPath, err)
		// If model fails to load, the network is already initialized by NewNetwork
	} else {
		fmt.Println("Model loaded successfully.")
	}
}

func main() {
	http.HandleFunc("/", serveStatic)
	http.HandleFunc("/predict", predictHandler)

	fmt.Printf("Server starting on port %s\n", port)
	log.Fatal(http.ListenAndServe(port, nil))
}

func serveStatic(w http.ResponseWriter, r *http.Request) {
	// Serve index.html for root requests, otherwise static files
	path := r.URL.Path
	if path == "/" {
		path = "/index.html"
	}
	http.ServeFile(w, r, filepath.Join("static", path))
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
		return
	}

	var requestData struct {
		Image string `json:"image"` // Base64 encoded or comma-separated string
	}

	err := json.NewDecoder(r.Body).Decode(&requestData)
	if err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Parse the comma-separated image data string into a float64 slice
	pixelsStr := strings.Split(requestData.Image, ",")
	if len(pixelsStr) != inputSize {
		http.Error(w, fmt.Sprintf("Expected %d pixels, got %d", inputSize, len(pixelsStr)), http.StatusBadRequest)
		return
	}

	inputPixels := make([]float64, inputSize)
	for i, s := range pixelsStr {
		val, err := strconv.ParseFloat(s, 64)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid pixel value '%s': %v", s, err), http.StatusBadRequest)
			return
		}
		inputPixels[i] = val // Pixels are expected to be normalized 0.0-1.0
	}

	inputMatrix := matrix.NewMatrix(1, inputSize)
	// Apply Center of Mass centering
	centeredPixels := centerImage(inputPixels)
	inputMatrix[0] = centeredPixels

	// Debug: Print input image as ASCII art
	fmt.Println("--- Processed (Centered) Image Input ---")
	for r := 0; r < 28; r++ {
		for c := 0; c < 28; c++ {
			val := centeredPixels[r*28+c]
			if val > 0.5 {
				fmt.Print("#") // Ink
			} else if val > 0.1 {
				fmt.Print(".") // Faint ink
			} else {
				fmt.Print(" ") // Background
			}
		}
		fmt.Println()
	}
	fmt.Println("----------------------------------------")

	prediction, err := net.Predict(inputMatrix)
	if err != nil {
		http.Error(w, fmt.Sprintf("Prediction failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]int{"prediction": prediction}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// centerImage shifts the image so that its center of mass is at (14, 14).
func centerImage(pixels []float64) []float64 {
	sumX, sumY, totalWeight := 0.0, 0.0, 0.0
	for i, val := range pixels {
		if val > 0 {
			x := float64(i % 28)
			y := float64(i / 28)
			sumX += x * val
			sumY += y * val
			totalWeight += val
		}
	}

	if totalWeight == 0 {
		return pixels
	}

	centerX := sumX / totalWeight
	centerY := sumY / totalWeight

	shiftX := 14.0 - centerX
	shiftY := 14.0 - centerY

	newPixels := make([]float64, len(pixels))
	for r := 0; r < 28; r++ {
		for c := 0; c < 28; c++ {
			// Find which source pixel maps to (c, r)
			// (c, r) = (srcX + shiftX, srcY + shiftY)
			// => srcX = c - shiftX
			srcX := int(float64(c) - shiftX)
			srcY := int(float64(r) - shiftY)

			if srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28 {
				newPixels[r*28+c] = pixels[srcY*28+srcX]
			}
		}
	}
	return newPixels
}
