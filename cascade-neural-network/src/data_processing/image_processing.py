import cv2
import numpy as np

def process_dataset(test=False):
    """
    Load images and landmarks, preprocess them, and save cropped images and landmarks.

    Args:
        test (bool): Whether to load test data or training data.

    Returns:
        None
    """
    size = 105 if test else 500
    ctr = -1

    for shp in range(6):
        for i in range(size):
            ctr += 1
            o_landmark_path, o_image_path, landmark_path, image_path = get_file_paths(test, i, ctr)

            smallest_x, smallest_y, greatest_x, greatest_y = find_landmark_extents(o_landmark_path)

            temp_x, temp_y = process_landmarks(o_landmark_path, smallest_x, smallest_y)

            width, height = greatest_x - smallest_x, greatest_y - smallest_y

            temp_x, temp_y = normalize_landmarks(temp_x, temp_y, shp, width, height)

            Y = store_landmarks(Y, temp_x, temp_y, i)

            img = cv2.imread(o_image_path)
            crop_and_resize_image(img, smallest_x, smallest_y, greatest_x, greatest_y, shp, image_path)

            save_landmarks(landmark_path, Y, i)

def get_file_paths(test, i, ctr):
    """
    Generate file paths for input and output data files based on parameters.

    Args:
        test (bool): Whether to load test data or training data.
        i (int): Index of the current data item.
        ctr (int): Counter for file numbers.

    Returns:
        tuple: A tuple containing the paths for original landmark, original image, preprocessed landmark, and preprocessed image.
    """
    if test:
        prefix = 'data/test'
    else:
        prefix = 'data/train'

    o_landmark_path = f'{prefix}/o_landmarks/{prefix}_{i}.txt'
    o_image_path = f'{prefix}/o_images/{prefix}_{i}.png'
    landmark_path = f'{prefix}/landmarks/{prefix}_{ctr}.txt'
    image_path = f'{prefix}/images/{prefix}_{ctr}.png'

    return o_landmark_path, o_image_path, landmark_path, image_path

def find_landmark_extents(landmark_path):
    """
    Find the smallest and greatest X and Y coordinates in a landmark file.

    Args:
        landmark_path (str): Path to the landmark file.

    Returns:
        tuple: A tuple containing the smallest X, smallest Y, greatest X, and greatest Y coordinates.
    """
    smallest_x = 999999
    smallest_y = 999999
    greatest_x = 0
    greatest_y = 0

    with open(landmark_path, 'r') as f:
        lines_list = f.readlines()

        for j in range(3, 58):
            string = lines_list[j]
            str1, str2 = string.split(' ')
            x_ = float(str1)
            y_ = float(str2)

            smallest_x = min(smallest_x, int(x_))
            greatest_x = max(greatest_x, int(x_))
            smallest_y = min(smallest_y, int(y_))
            greatest_y = max(greatest_y, int(y_))

    if smallest_x > 5 and smallest_y > 5:
        smallest_x -= 5
        smallest_y -= 5
    else:
        smallest_x = 0
        smallest_y = 0
    greatest_x += 5
    greatest_y += 5

    return smallest_x, smallest_y, greatest_x, greatest_y

def process_landmarks(o_landmark_path, smallest_x, smallest_y):
    """
    Process landmark coordinates by adjusting for the smallest X and Y values.

    Args:
        o_landmark_path (str): Path to the original landmark file.
        smallest_x (int): Smallest X coordinate value.
        smallest_y (int): Smallest Y coordinate value.

    Returns:
        tuple: A tuple containing the processed X and Y coordinates.
    """
    temp_x = np.zeros(55)
    temp_y = np.zeros(55)

    with open(o_landmark_path, 'r') as f:
        lines_list = f.readlines()

        for j in range(3, 58):
            string = lines_list[j]
            str1, str2 = string.split(' ')
            x_ = float(str1)
            y_ = float(str2)

            x_ -= smallest_x
            y_ -= smallest_y

            temp_x[j - 3] = round(x_, 3)
            temp_y[j - 3] = round(y_, 3)

    return temp_x, temp_y

def normalize_landmarks(temp_x, temp_y, shp, width, height):
    """
    Normalize landmark coordinates based on image dimensions and transformation shape.

    Args:
        temp_x (numpy.ndarray): X coordinates of landmarks.
        temp_y (numpy.ndarray): Y coordinates of landmarks.
        shp (int): Transformation shape identifier.
        width (int): Width of the bounding box.
        height (int): Height of the bounding box.

    Returns:
        tuple: A tuple containing the normalized X and Y coordinates.
    """
    for k in range(55):
        temp_x[k] /= width
        temp_y[k] /= height

        if shp == 1:
            temp_x[k] = 1 - temp_x[k]

        elif shp == 2:
            temp_x[k], temp_y[k] = temp_y[k], 1 - temp_x[k]

        elif shp == 3:
            temp_x[k], temp_y[k] = 1 - temp_y[k], temp_x[k]

        elif shp == 4:
            temp_x[k], temp_y[k] = 1 - temp_x[k], 1 - temp_y[k]

        elif shp == 5:
            temp_x[k], temp_y[k] = 1 - temp_x[k], 1 - temp_y[k]

    return temp_x, temp_y

def store_landmarks(Y, temp_x, temp_y, i):
    """
    Store landmarks for each data item, appending to the existing landmark array.

    Args:
        Y (numpy.ndarray): Array to store landmark coordinates.
        temp_x (numpy.ndarray): X coordinates of landmarks.
        temp_y (numpy.ndarray): Y coordinates of landmarks.
        i (int): Index of the current data item.

    Returns:
        numpy.ndarray: Updated array with appended landmarks.
    """
    if i == 0:
        Y = np.hstack((temp_x, temp_y))
        Y = Y[None, :]
    else:
        temp = np.hstack((temp_x, temp_y))
        temp = temp[None, :]
        Y = np.vstack((Y, temp))

    return Y

def crop_and_resize_image(img, smallest_x, smallest_y, greatest_x, greatest_y, shp, image_path):
    """
    Crop and resize an image based on landmark extents and transformation shape.

    Args:
        img (numpy.ndarray): Input image.
        smallest_x (int): Smallest X coordinate value.
        smallest_y (int): Smallest Y coordinate value.
        greatest_x (int): Greatest X coordinate value.
        greatest_y (int): Greatest Y coordinate value.
        shp (int): Transformation shape identifier.
        image_path (str): Path to save the processed image.

    Returns:
        None
    """
    cv2.rectangle(img, (smallest_x, smallest_y), (greatest_x, greatest_y), (255, 255, 255), 3)
    crop_image = img[smallest_y:greatest_y, smallest_x:greatest_x]
    resize_image = cv2.resize(crop_image, (224, 224))

    if shp == 1:
        resize_image = cv2.flip(resize_image, 1)

    elif shp == 2:
        resize_image = np.rot90(resize_image)

    elif shp == 3:
        resize_image = np.rot90(resize_image, 3)

    elif shp == 4:
        resize_image = cv2.flip(resize_image, 1)
        resize_image = np.rot90(resize_image)

    elif shp == 5:
        resize_image = cv2.flip(resize_image, 1)
        resize_image = np.rot90(resize_image, 3)

    cv2.imwrite(image_path, resize_image)

def save_landmarks(landmark_path, Y, i):
    """
    Save processed landmarks to a file.

    Args:
        landmark_path (str): Path to save the landmark file.
        Y (numpy.ndarray): Array containing landmark coordinates.
        i (int): Index of the current data item.

    Returns:
        None
    """
    with open(landmark_path, 'w') as f:
        f.write("version: 2\n")
        f.write("n_points: 55\n")
        f.write("{\n")
        for p in range(55):
            line = str(Y[i][p]) + " " + str(Y[i][p + 55]) + "\n"
            f.write(line)
        f.write("}\n")
