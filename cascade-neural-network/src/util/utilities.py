import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def load_single_image(single_img_path):
    """
    Load and preprocess a single image for the model.

    Args:
        single_img_path (str): Path to the single image file.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    img = cv2.imread(single_img_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite('data/single/single_img.png', img)
    img = image.load_img('data/single/single_img.png')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_multiple_images(size, test=False):
    """
    Load and preprocess multiple images for the model.

    Args:
        size (int): Number of images to load.
        test (bool): Whether to load test data.

    Returns:
        numpy.ndarray: Preprocessed images.
    """
    if test:
        img_paths = [f'data/test/images/test_{i}.png' for i in range(size)]
    else:
        img_paths = [f'data/train/images/train_{i}.png' for i in range(size)]

    X = None
    for img_path in img_paths:
        img = image.load_img(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
    return X

def load_landmarks(size, test=False):
    """
    Load landmarks for multiple images.

    Args:
        size (int): Number of landmarks to load.
        test (bool): Whether to load test data.

    Returns:
        numpy.ndarray: Landmarks for the loaded images.
    """
    if test:
        txt_paths = [f'data/test/landmarks/test_{i}.txt' for i in range(size)]
    else:
        txt_paths = [f'data/train/landmarks/train_{i}.txt' for i in range(size)]

    Y = None
    for txt_path in txt_paths:
        temp_x, temp_y = load_landmark_from_file(txt_path)
        if Y is None:
            Y = np.hstack((temp_x, temp_y))[None, :]
        else:
            temp = np.hstack((temp_x, temp_y))[None, :]
            Y = np.vstack((Y, temp))
    return Y

def load_landmark_from_file(txt_path):
    """
    Load landmarks from a single landmark file.

    Args:
        txt_path (str): Path to the landmark file.

    Returns:
        tuple: X and Y coordinates of the landmarks.
    """
    with open(txt_path, 'r') as f:
        lines_list = f.readlines()

    temp_x, temp_y = None, None
    for j in range(3, 58):
        string = lines_list[j]
        str1, str2 = string.split(' ')
        x_ = float(str1)
        y_ = float(str2)

        if j == 3:
            temp_x = np.array(x_)
            temp_y = np.array(y_)
        else:
            temp_x = np.hstack((temp_x, x_))
            temp_y = np.hstack((temp_y, y_))

    return temp_x, temp_y

def load_data(test=False, size=3000, single_img=False, single_img_path='give a path please'):
    """
    Load and preprocess data (images and landmarks).

    Args:
        test (bool): Whether to load test data.
        size (int): Number of images or landmarks to load.
        single_img (bool): Whether to load a single image.
        single_img_path (str): Path to the single image file.

    Returns:
        tuple: Preprocessed images and landmarks (X, Y).
    """
    if single_img:
        return load_single_image(single_img_path), None
    else:
        X = load_multiple_images(size, test)
        Y = load_landmarks(size, test)
        return X, Y
