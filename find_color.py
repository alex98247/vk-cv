import os
import tensorflow_hub as hub
import tensorflow as tf
import cv2

from calc_metric import calc_metric

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
colors = ['black', 'blue', 'green', 'red', 'white_silver', 'yellow']
delta = 20


def detect_color(r, g, b):
    min_color = min(r, g, b)
    max_delta = max(abs(r - g), abs(r - b), abs(g - b))
    if max_delta < delta and min_color < 80:
        return 'black'
    elif abs(r - g) < delta and abs(r - b) < delta and abs(g - b) < delta:
        return 'white_silver'
    elif abs(b - g) < 50 and b - r > 100 and g - r > 100:
        return 'blue_cyan'
    elif abs(r - g) < 50 and g - b > 100 and r - b > 100:
        return 'yellow'
    elif g > r and g > b:
        return 'green'
    else:
        return 'red'


def find_color(input_dir, output_file='output_color.csv'):
    filenames = next(os.walk(input_dir), (None, None, []))[2]
    result = open(output_file, "w")

    for filename in filenames:
        img = cv2.imread(os.path.join(input_dir, filename))
        inp = cv2.resize(img, (1028, 1028))
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

        rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
        rgb_tensor = tf.expand_dims(rgb_tensor, 0)
        boxes, scores, classes, num_detections = detector(rgb_tensor)

        pred_boxes = boxes.numpy()[0].astype('int')
        pred_scores = scores.numpy()[0]

        for score, (ymin, xmin, ymax, xmax) in zip(pred_scores, pred_boxes):
            if score < 0.5:
                continue
            #img_boxes = cv2.rectangle(rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
            r, g, b = calc_metric(rgb, xmin, ymin, xmax - xmin, ymax - ymin)
            print(filename, r, g, b, detect_color(r, g, b))
            result.write(filename + ',' + detect_color(r, g, b) + '\n')
            break
    result.close()


find_color('dataset/data2')
