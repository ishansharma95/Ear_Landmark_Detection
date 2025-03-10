from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense

def create_cnn_model():
    """
    Create a CNN model for ear landmark detection.

    Returns:
        keras.models.Sequential: The CNN model.
    """
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3), kernel_initializer='random_uniform', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(110))
    return model

def compile_and_train_model(model, X_train, Y_train, optimizer, loss, metrics, epochs, batch_size):
    """
    Compile and train the CNN model.

    Args:
        model (keras.models.Sequential): The CNN model.
        X_train (numpy.ndarray): Training images.
        Y_train (numpy.ndarray): Training landmarks.
        optimizer (str): Name of the optimizer.
        loss (str): Name of the loss function.
        metrics (list): List of evaluation metrics.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        keras.callbacks.History: Training history.
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

def save_trained_model(model, file_name):
    """
    Save the trained model to a file.

    Args:
        model (keras.models.Sequential): The trained CNN model.
        file_name (str): Name of the file to save the model.
    """
    model.save(file_name + '.h5')

def load_saved_model(file_name):
    """
    Load a saved model from a file.

    Args:
        file_name (str): Name of the saved model file.

    Returns:
        keras.models.Sequential: The loaded model.
    """
    return load_model(file_name + '.h5')

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model on test data and print the loss and accuracy.

    Args:
        model (keras.models.Sequential): The trained CNN model.
        X_test (numpy.ndarray): Test images.
        Y_test (numpy.ndarray): Test landmarks.
    """
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

def summarize_model_architecture(model):
    """
    Print a summary of the model architecture.

    Args:
        model (keras.models.Sequential): The CNN model.
    """
    model.summary()
