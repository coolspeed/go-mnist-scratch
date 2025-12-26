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
	inputMatrix[0] = inputPixels

	prediction, err := net.Predict(inputMatrix)
	if err != nil {
		http.Error(w, fmt.Sprintf("Prediction failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]int{"prediction": prediction}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
