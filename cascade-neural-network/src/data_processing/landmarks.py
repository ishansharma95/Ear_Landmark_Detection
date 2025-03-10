import matplotlib.pyplot as plt

def put_landmarks(i, pred, single_img=False):
    """
    Place landmark points on an image and save the result.

    Parameters:
        i (int): Index or identifier for the image.
        pred (numpy.ndarray): Predicted landmark points for the image.
        single_img (bool, optional): Flag indicating if a single image is processed.
            If True, uses default single image paths; otherwise, uses indexed paths. Default is False.

    Returns:
        None

    Raises:
        FileNotFoundError: If image paths are not found.

    This function takes an index or identifier 'i' and a set of predicted landmark points 'pred' for an image.
    It then places the landmark points on the image and saves the result to a file.

    If 'single_img' is True, it uses default paths for a single sample image and result.
    If 'single_img' is False (default), it constructs image paths based on the index 'i' for a whole set of images.

    Example usage:
    For a single image:
    put_landmarks(0, predicted_landmarks, single_img=True)

    For a set of images:
    put_landmarks(1, predicted_landmarks)
    """
    img_path = 'data/test/images/test_' + str(i) + '.png'
    img_result_path = 'data/test/results/result_' + str(i) + '.png'

    if single_img:
        img_path = 'data/single/sampleimage.png'
        img_result_path = 'data/single/result/result.png'

    img_original = plt.imread(img_path)

    # Place the landmark points on the image
    for j in range(0, 55):
        plt.scatter([pred[j]], [pred[j + 55]])

    # Display the original image with landmark points and save the result
    plt.imshow(img_original)
    plt.savefig(img_result_path)
    plt.close()
