import os
import re
import cv2


def get_image_count():
    with open('dataset/image_counter.txt', 'r') as fp:
        return int(fp.read())


def remove_channel(filename):
    return re.sub('_[r|g|b]\.jpg', '', filename)


def merge_channels(input_dir, output_dir):
    image_count = get_image_count()
    filenames = next(os.walk(input_dir), (None, None, []))[2]

    # Set may reorder files. Task does not requires ordering
    filenames = list(set(map(remove_channel, filenames)))[:image_count]
    for filename in filenames:
        b = cv2.imread(os.path.join(input_dir, filename + '_b.jpg'), cv2.COLOR_BGR2GRAY)
        g = cv2.imread(os.path.join(input_dir, filename + '_g.jpg'), cv2.COLOR_BGR2GRAY)
        r = cv2.imread(os.path.join(input_dir, filename + '_r.jpg'), cv2.COLOR_BGR2GRAY)
        colored_image = cv2.merge((b, g, r))
        cv2.imwrite(os.path.join(output_dir, filename + '.jpg'), colored_image)
