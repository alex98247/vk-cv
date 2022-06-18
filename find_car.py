import os
import tensorflow_hub as hub
import tensorflow as tf
import cv2

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
cars = [3, 6, 8]

def find_car(input_dir, output_cars='output.csv'):
    filenames = next(os.walk(input_dir), (None, None, []))[2]
    result = open(output_cars, "w")

    for filename in filenames:
        img = cv2.imread(os.path.join(input_dir, filename))
        inp = cv2.resize(img, (1028, 1028))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_labels = classes.numpy().astype('int')[0]
        pred_scores = scores.numpy()[0]

        is_car = False
        for score, label in zip(pred_scores, pred_labels):
            if score > 0.5 and label in cars:
                is_car = True
                break
        result.write(filename + ',' + str(is_car) + '\n')
    result.close()
