import os
import tensorflow as tf
from object_detection.utils import dataset_util
from glob import glob

# Update this
label_map = {"mosquito": 1}
input_image_dir = "D:/mosquito_tf_dataset/images"
input_label_dir = "D:/mosquito_tf_dataset/labels"
output_path = "D:/mosquito_tf_dataset"

splits = ["train", "val"]

def create_tf_example(img_path, label_path):
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    filename = os.path.basename(img_path).encode('utf8')
    image_format = b'jpg'

    image = tf.io.decode_jpeg(encoded_jpg)
    height, width = image.shape[0], image.shape[1]

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, w, h = map(float, parts)
            x_min = max((x - w / 2), 0)
            x_max = min((x + w / 2), 1)
            y_min = max((y - h / 2), 0)
            y_max = min((y + h / 2), 1)

            xmins.append(x_min)
            xmaxs.append(x_max)
            ymins.append(y_min)
            ymaxs.append(y_max)
            classes_text.append(b'mosquito')
            classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

for split in splits:
    writer = tf.io.TFRecordWriter(os.path.join(output_path, f"{split}.record"))
    img_dir = os.path.join(input_image_dir, split)
    lbl_dir = os.path.join(input_label_dir, split)
    image_paths = glob(os.path.join(img_dir, "*.jpg"))
    for img_path in image_paths:
        label_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if not os.path.exists(label_path):
            continue
        tf_example = create_tf_example(img_path, label_path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f"✅ TFRecord for {split} saved.")
