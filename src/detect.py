"""Main script to run the object detection routine."""

import argparse
import sys
import time

import RPi.GPIO as GPIO

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=5,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)
  # Set GPIO pin-scheme
  GPIO.setmode(GPIO.BCM)
  GPIO.setwarnings(False)
  
  # Set pin modes
  GPIO.setup(14, GPIO.OUT)
  GPIO.setup(15, GPIO.OUT)
  GPIO.setup(18, GPIO.OUT)
  GPIO.setup(23, GPIO.OUT)
  

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )


    counter += 1

    # Run object detection estimation using the model.
    detections = detector.detect(image)

    # Draw keypoints and edges on input image
    image, rg = utils.visualize(image, detections)
    r1, r2, r3, r4 = rg.r1, rg.r2, rg.r3, rg.r4

    for i in (14, 15, 18, 23):
      GPIO.output(i, GPIO.LOW)
    #GPIO.output(15, GPIO.LOW)
    #GPIO.output(18, GPIO.LOW)
    #GPIO.output(23, GPIO.LOW)
    
    if r1:
        GPIO.output(14, GPIO.HIGH)
    if r2:
        GPIO.output(15, GPIO.HIGH)
    if r3:
        GPIO.output(18, GPIO.HIGH)
    if r4:
        GPIO.output(23, GPIO.HIGH)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('Aid for Visually Impaired', image)

  cap.release()
  cv2.destroyAllWindows()
  GPIO.cleanup()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
