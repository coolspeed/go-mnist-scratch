package utils

import (
	"encoding/binary"
	"math"
	"os"
)

const (
	imageMagic = 2051
	labelMagic = 2049
)

type MNISTData struct {
	Images []*Image
	Labels []*Label
}

type Image struct {
	Pixels [][]float64
}

type Label struct {
	Value  int
	OneHot []float64
}

func LoadImages(filename string) ([]*Image, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, numImages, rows, cols int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &numImages)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	if magic != imageMagic {
		return nil, os.ErrInvalid
	}

	images := make([]*Image, numImages)
	for i := 0; i < int(numImages); i++ {
		pixels := make([]byte, rows*cols)
		file.Read(pixels)

		image := &Image{
			Pixels: make([][]float64, rows),
		}
		for r := 0; r < int(rows); r++ {
			image.Pixels[r] = make([]float64, cols)
			for c := 0; c < int(cols); c++ {
				image.Pixels[r][c] = float64(pixels[r*int(cols)+c]) / 255.0
			}
		}
		images[i] = image
	}

	return images, nil
}

func LoadLabels(filename string) ([]*Label, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, numLabels int32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &numLabels)

	if magic != labelMagic {
		return nil, os.ErrInvalid
	}

	labels := make([]*Label, numLabels)
	for i := 0; i < int(numLabels); i++ {
		var label byte
		binary.Read(file, binary.BigEndian, &label)

		oneHot := make([]float64, 10)
		oneHot[int(label)] = 1.0

		labels[i] = &Label{
			Value:  int(label),
			OneHot: oneHot,
		}
	}

	return labels, nil
}

func LoadMNIST(imageFile, labelFile string) (*MNISTData, error) {
	images, err := LoadImages(imageFile)
	if err != nil {
		return nil, err
	}

	labels, err := LoadLabels(labelFile)
	if err != nil {
		return nil, err
	}

	return &MNISTData{
		Images: images,
		Labels: labels,
	}, nil
}

func FlattenImage(image *Image) []float64 {
	flattened := make([]float64, len(image.Pixels)*len(image.Pixels[0]))
	for i, row := range image.Pixels {
		for j, pixel := range row {
			flattened[i*len(row)+j] = pixel
		}
	}
	return flattened
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return Sigmoid(x) * (1.0 - Sigmoid(x))
}

func Softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, val := range x {
		if val > maxVal {
			maxVal = val
		}
	}

	expVals := make([]float64, len(x))
	sum := 0.0
	for i, val := range x {
		expVals[i] = math.Exp(val - maxVal)
		sum += expVals[i]
	}

	for i := range expVals {
		expVals[i] /= sum
	}

	return expVals
}
