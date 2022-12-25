import os

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import cv2 
import numpy as np
from PIL import ImageGrab

import keys as k

keys = k.Keys({})
def send_mouse_input(dx, dy):
    keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0001, dx, dy))

model_name = "my_ssd_mobnet"
min_detection_confidence = 0.8
mouse_move_coeff = 0.33

#configs = config_util.get_configs_from_pipeline_file(pipeline_config)

src_dir = os.path.dirname(os.path.dirname(__file__))
models_dir = os.path.join(src_dir, "models")
annotations_dir = os.path.join(src_dir, "annotations")

model_path = os.path.join(models_dir, model_name)
label_map_file_path = os.path.join(annotations_dir, "label_map.pbtxt")

detection_classes = ['enemy']


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections

  return detect_fn


if __name__ == "__main__":
    # Load Model
    pipeline_config_path = os.path.join(model_path, "pipeline.config")
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']

    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_path, 'ckpt-51')).expect_partial()

    # Get the detection function for this model
    model_detect_fn = get_model_detection_function(detection_model)

    # Create the category index
    category_index = label_map_util.create_category_index_from_labelmap(label_map_file_path)
    
    # cap = cv2.VideoCapture("test.mp4")
    # frame_width = 1920
    # frame_height = 1080
    frame_width = 640
    frame_height = 480
    screen_center_point = (int(frame_width/2), int(frame_height/2))

    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()

    while True:
        # ret, image_np = cap.read()
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break

        image_np = np.array(ImageGrab.grab(bbox=(0, 0, 640, 480)))

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = model_detect_fn(input_tensor)

        # print(detections['detection_scores'])
        # print(detections['detection_boxes'])
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

        aim_points = []

        for i in range(0, len(detections['detection_scores'])):
            if (detections['detection_scores'][i] >= min_detection_confidence):
                ymin = detections['detection_boxes'][i][0]
                xmin = detections['detection_boxes'][i][1]
                ymax = detections['detection_boxes'][i][2]
                xmax = detections['detection_boxes'][i][3]

                (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width, 
                              ymin * frame_height, ymax * frame_height)

                bbox_center = (int((left + right) / 2), int(top - ((top - bottom)/4)))
                aim_points.append(bbox_center)                

        print(aim_points)

        closest_aim_point_manhattan_dist = 5*frame_width
        closest_aim_point_dist_dx = 0
        closest_aim_point_dist_dy = 0

        for aim_point in aim_points:
            dx = aim_point[0] - screen_center_point[0]
            dy = aim_point[1] -screen_center_point[1]
            manhattan_dist = dx + dy

            if manhattan_dist < closest_aim_point_manhattan_dist:
                closest_aim_point_dist_dx = int(dx * mouse_move_coeff)
                closest_aim_point_dist_dy = int(dy * mouse_move_coeff)

            image_np = cv2.line(image_np, screen_center_point, aim_point, (0, 255, 0), 5)

        cv2.imshow('object detection',  cv2.resize(image_np, (640, 360)))
        print("dist = ", end = " ")
        print((closest_aim_point_dist_dx, closest_aim_point_dist_dy))

        # Move mouse to centre of nearest detection

        # If mouse already in nearest detection, shoot

        send_mouse_input(closest_aim_point_dist_dx, closest_aim_point_dist_dy)

        if (closest_aim_point_dist_dx <= 10 and closest_aim_point_dist_dy <= 10
        and closest_aim_point_dist_dx > 0 and closest_aim_point_dist_dy > 0) :
            keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0002))
            keys.keys_worker.SendInput(keys.keys_worker.Mouse(0x0004))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cap.release()
            break


    pass

