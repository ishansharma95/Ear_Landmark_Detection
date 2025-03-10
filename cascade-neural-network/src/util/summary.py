from src.model.landmark_detection_cnn_model import load_saved_model, summarize_model_architecture

def main():
    # Load the model
    model = load_saved_model('trained_model')

    # Summarize the model
    summarize_model_architecture(model)

    # Access and print the learning rate from the optimizer
    learning_rate = model.optimizer.lr
    print(f"Learning rate: {learning_rate}")

if __name__ == "__main__":
    main()