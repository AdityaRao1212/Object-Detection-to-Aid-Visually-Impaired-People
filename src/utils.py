"""Utility functions to display the pose detection results."""

from typing import List, NamedTuple

import cv2
import numpy as np
from object_detector import Detection
import random
_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 2
_FONT_THICKNESS = 2
_TEXT_COLOR = (40, 200, 0)  # green
_PARTITION_COLOR = (0, 255, 255)  # yellow


class vibrateLocations():
    def __init__(self):
        self.r1: bool = False
        self.r2: bool = False
        self.r3: bool = False
        self.r4: bool = False


def getColor() -> tuple:
    color = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 0, 255)
    }
    out = random.randint(1, 4)
    return color[out]


def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> tuple:
    """Draws bounding boxes on the input image and return it.

    Args:
      image: The input RGB image.
      detections: The list of all "Detection" entities to be visualize.

    Returns:
      Image with bounding boxes.
    """
    # Initialise vibrate Locations object
    imgRegions = vibrateLocations()
    # Draw partition lines
    height, width, _ = image.shape
    w = int(width/4)
    start = (w, 0)
    end = (w, 480)
    for i in range(2, 5):
        image = cv2.line(image, start, end, _PARTITION_COLOR, 1, cv2.LINE_AA)
        start = (w*i, 0)
        end = (w*i, 480)

    for detection in detections:
        # Draw bounding_box
        left, right = detection.bounding_box.left, detection.bounding_box.right
        start_point = left, detection.bounding_box.top
        end_point = right, detection.bounding_box.bottom
        color = getColor()
        cv2.rectangle(image, start_point, end_point, color, 2)
        if (left < w):
            imgRegions.r1 = True
        elif left < (w*2):
            imgRegions.r2 = True
        elif left < (w*3):
            imgRegions.r3 = True
        elif left < (w*4):
            imgRegions.r4 = True

        # Draw label and score
        category = detection.categories[0]
        class_name = category.label
        probability = round(category.score, 2)
        result_text = class_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + detection.bounding_box.left,
                         _MARGIN + _ROW_SIZE + detection.bounding_box.top)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    return (image, imgRegions)
