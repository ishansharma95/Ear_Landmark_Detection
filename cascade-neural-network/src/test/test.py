from src.model.landmark_detection_cnn_model import load_current_model
from src.util.utilities import load_data
from src.data_processing.landmarks import put_landmarks

def predict_landmarks_for_image(model, image):
    """
    Predict landmark points on an image using a given model.

    Parameters:
        model: A pre-trained landmark detection model.
        image (numpy.ndarray): The input image for landmark prediction.

    Returns:
        numpy.ndarray: Predicted landmark points for the image.
    """
    image = image[None, :]  # Adjust the dimensions for the model
    prediction = model.predict(image)

    for i in range(len(prediction[0])):
        prediction[0][i] = int(prediction[0][i] * 224)  # Adjust landmark points for a 224x224 image

    return prediction[0]

def process_and_display_landmarks_for_images(model, image_paths, single_img=False):
    """
    Load a pre-trained model, predict landmarks for a list of images, and display them on the images.

    Parameters:
        model: A pre-trained landmark detection model.
        image_paths (list): List of paths to the test images.
        single_img (bool): Flag indicating if a single image is used.

    Returns:
        None
    """
    for i, image_path in enumerate(image_paths):
        image = load_data(test=True, test_size=630, single_img=single_img, single_img_path=image_path)
        landmarks = predict_landmarks_for_image(model, image)
        put_landmarks(i, landmarks, single_img=single_img)

if __name__ == "__main__":
    model_path = 'trained_model'
    model = load_current_model(model_path)
    image_paths = ['data/single/single_img.png']  # Add more paths if needed
    single_img = False  # Set to True if using a single image
    process_and_display_landmarks_for_images(model, image_paths, single_img)
