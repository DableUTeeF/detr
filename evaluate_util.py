import numpy as np
from six import raise_from
import csv
import sys
import cv2

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-faster_rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-faster_rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-faster_rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


def all_annotation_from_instance(instance):
    all_annotation = [[], []]
    for obj in instance['object']:
        if obj['name'] == 'ov':
            all_annotation[0].append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
        else:
            all_annotation[1].append(np.array([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]))
    return np.array(all_annotation)


def evaluate(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    # all_detections = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]  # [[bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)], [bbox(x1, y1, x2, y2), bbox(x1, y1, x2, y2)]]
    # all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(generator.size())]
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    for label in range(num_classes):
        print()
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            print(i, end='\r')
            detections = all_detections[i][label]
            annotations = np.array(all_annotations[i][label])
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision
        total_instances.append(num_annotations)

    return average_precisions, total_instances


def get_false_positive(all_detections, all_annotations, num_classes, iou_threshold=0.5):
    assert len(all_annotations) == len(all_detections)
    average_precisions = {}
    total_instances = []
    false_positives = []
    for i in range(len(all_annotations)):
        print(i, end='\r')
        detections = all_detections[i]
        annotations = np.array(all_annotations[i])
        detected_annotations = []

        for d in detections:

            if annotations.shape[0] == 0:
                continue

            overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                pass
            else:
                false_positives.append(d)
    return false_positives


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                       None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def create_csv_training_instances(train_csv, test_csv, class_csv, with_wh=False):
    with _open_for_csv(class_csv) as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    with _open_for_csv(train_csv) as file:
        train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    with _open_for_csv(test_csv) as file:
        test_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    train_ints = []
    valid_ints = []
    labels = list(classes)
    max_box_per_image = 0
    for k in train_image_data:
        image_data = train_image_data[k]
        ints = {'filename': k, 'object': []}
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            if with_wh:
                x = cv2.imread(k)
                height, width, _ = x.shape
                o['width'] = width
                o['height'] = height
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        train_ints.append(ints)

    for k in test_image_data:
        image_data = test_image_data[k]
        ints = {'filename': k, 'object': []}
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            if with_wh:
                x = cv2.imread(k)
                height, width, _ = x.shape
                o['width'] = width
                o['height'] = height
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        valid_ints.append(ints)

    return train_ints, valid_ints, sorted(labels), max_box_per_image

