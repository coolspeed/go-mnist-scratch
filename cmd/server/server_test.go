package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coolspeed/go-mnist-scratch/neural"
)

func TestPredictHandler(t *testing.T) {
	// Initialize the global network variable for testing
	// We use a small dummy network to avoid loading large model files during unit tests
	net = neural.NewNetwork(784, 10, 10, 0.1)

	// Create a dummy 28x28 image (comma-separated string of 784 zeros)
	dummyImage := strings.Repeat("0.0,", 783) + "0.0"
	requestBody, _ := json.Marshal(map[string]string{
		"image": dummyImage,
	})

	req, err := http.NewRequest("POST", "/predict", bytes.NewBuffer(requestBody))
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(predictHandler)

	handler.ServeHTTP(rr, req)

	// Check status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("handler returned wrong status code: got %v want %v",
			status, http.StatusOK)
	}

	// Check response body structure
	var response map[string]interface{}
	if err := json.NewDecoder(rr.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Verify 'prediction' exists
	if _, ok := response["prediction"]; !ok {
		t.Error("Response missing 'prediction' field")
	}

	// Verify 'duration_us' exists and is valid
	duration, ok := response["duration_us"]
	if !ok {
		t.Error("Response missing 'duration_us' field")
	} else {
		// JSON numbers are float64 by default in generic maps
		durationVal, ok := duration.(float64)
		if !ok {
			t.Errorf("duration_us is not a number, got %T", duration)
		}
		if durationVal < 0 {
			t.Errorf("duration_us should be positive, got %v", durationVal)
		}
	}
}
