.PHONY: build train server validate test clean help

# Binary names
SERVER_BIN=bin/server
TRAIN_BIN=bin/train
VALIDATE_BIN=bin/validate

help:
	@echo "Usage:"
	@echo "  make build      - Build the inference server"
	@echo "  make train      - Run the training process"
	@echo "  make server     - Build and run the inference server"
	@echo "  make validate   - Run the accuracy validation tool"
	@echo "  make test       - Run all unit tests"
	@echo "  make clean      - Remove built binaries and logs"

build:
	@echo "Building inference server..."
	go build -o $(SERVER_BIN) cmd/server/main.go
	@echo "Done. Run ./$(SERVER_BIN) to start the server."

train:
	@echo "Starting training process..."
	go run cmd/train/main.go

server: build
	./$(SERVER_BIN)

run: server

validate:
	@echo "Running validation..."
	go run cmd/validate/main.go

test:
	@echo "Running tests..."
	go test ./...

clean:
	@echo "Cleaning up..."
	rm -f $(SERVER_BIN) $(TRAIN_BIN) $(VALIDATE_BIN) server.log
	@echo "Clean complete."
