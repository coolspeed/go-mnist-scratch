package utils

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// ImageData holds the MNIST image data
type ImageData struct {
	MagicNumber uint32
	NumImages   uint32
	NumRows     uint32
	NumCols     uint32
	Images      [][]float64 // Normalized pixel data
}

// LabelData holds the MNIST label data
type LabelData struct {
	MagicNumber uint32
	NumLabels   uint32
	Labels      []uint8 // Raw labels
}

// LoadImages reads MNIST image data from a gzipped IDX file.
func LoadImages(filePath string) (*ImageData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader for image file: %w", err)
	}
	defer gz.Close()

	var magicNumber uint32
	if err := binary.Read(gz, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("failed to read magic number from image file: %w", err)
	}

	var numImages, numRows, numCols uint32
	if err := binary.Read(gz, binary.BigEndian, &numImages); err != nil {
		return nil, fmt.Errorf("failed to read number of images: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &numRows); err != nil {
		return nil, fmt.Errorf("failed to read number of rows: %w", err)
	}
	if err := binary.Read(gz, binary.BigEndian, &numCols); err != nil {
		return nil, fmt.Errorf("failed to read number of columns: %w", err)
	}

	totalPixels := int(numRows * numCols)
	images := make([][]float64, numImages)
	pixelBuffer := make([]byte, totalPixels)

	for i := uint32(0); i < numImages; i++ {
		_, err := io.ReadFull(gz, pixelBuffer)
		if err != nil {
			return nil, fmt.Errorf("failed to read pixels for image %d: %w", i, err)
		}

		images[i] = make([]float64, totalPixels)
		for j, pixel := range pixelBuffer {
			images[i][j] = float64(pixel) / 255.0 // Normalize to 0.0-1.0
		}
	}

	return &ImageData{
		MagicNumber: magicNumber,
		NumImages:   numImages,
		NumRows:     numRows,
		NumCols:     numCols,
		Images:      images,
	}, nil
}

// LoadLabels reads MNIST label data from a gzipped IDX file.
func LoadLabels(filePath string) (*LabelData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open label file: %w", err)
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader for label file: %w", err)
	}
	defer gz.Close()

	var magicNumber uint32
	if err := binary.Read(gz, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("failed to read magic number from label file: %w", err)
	}

	var numLabels uint32
	if err := binary.Read(gz, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read number of labels: %w", err)
	}

	labels := make([]uint8, numLabels)
	_, err = io.ReadFull(gz, labels)
	if err != nil {
		return nil, fmt.Errorf("failed to read labels: %w", err)
	}

	return &LabelData{
		MagicNumber: magicNumber,
		NumLabels:   numLabels,
		Labels:      labels,
	}, nil
}

// OneHotEncode converts a single digit label into a one-hot encoded vector.
func OneHotEncode(label uint8, numClasses int) []float64 {
	oneHot := make([]float64, numClasses)
	if int(label) < numClasses {
		oneHot[label] = 1.0
	}
	return oneHot
}

// LoadMNIST loads both image and label data, returning them in a structured format.
func LoadMNIST(imagePath, labelPath string) (*ImageData, *LabelData, error) {
	imageData, err := LoadImages(imagePath)
	if err != nil {
		return nil, nil, fmt.Errorf("error loading images: %w", err)
	}

	labelData, err := LoadLabels(labelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("error loading labels: %w", err)
	}

	return imageData, labelData, nil
}

// Helper to convert byte slice to uint32 (BigEndian)
func bytesToUint32(b []byte) uint32 {
	return binary.BigEndian.Uint32(b)
}

// This helper is not directly used but can be useful for debugging IDX parsing.
func printBytes(prefix string, b []byte) {
	fmt.Printf("%s: ", prefix)
	for _, val := range b {
		fmt.Printf("%02x ", val)
	}
	fmt.Println()
}

// Debug function to peek into gzip content (not for production use)
func peekGzipContent(filePath string, numBytes int) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gz.Close()

	buffer := make([]byte, numBytes)
	n, err := gz.Read(buffer)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read from gzip: %w", err)
	}

	fmt.Printf("First %d bytes of gzipped content from %s:\n", n, filePath)
	printBytes("Content", buffer[:n])
	return nil
}