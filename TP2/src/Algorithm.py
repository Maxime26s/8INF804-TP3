import cv2


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    return clahe_image


def resize(image, min=720):
    if image.shape[0] < image.shape[1]:
        new_width = int(min * image.shape[1] / image.shape[0])
        return cv2.resize(image, (new_width, min))
    else:
        new_height = int(min * image.shape[0] / image.shape[1])
        return cv2.resize(image, (min, new_height))
