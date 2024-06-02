import os
import json
import xml.etree.ElementTree as ET
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Affine, Blur
import cv2

import argparse



# Function to augment image and save it
def augment_and_save(image_path, augmentations, new_filename, bboxes, labels):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return None, None
    augmented = augmentations(image=image, bboxes=bboxes, labels=labels)
    augmented_image = augmented['image']
    cv2.imwrite(new_filename, augmented_image)
    return augmented['bboxes'], labels

# Function to update XML annotations
def update_xml(file_path, new_folder, bboxes, labels, iteration):
    tree = ET.parse(file_path)
    root = tree.getroot()

    for obj, bbox in zip(root.findall('object'), bboxes):
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(bbox[0]))
        bndbox.find('ymin').text = str(int(bbox[1]))
        bndbox.find('xmax').text = str(int(bbox[2]))
        bndbox.find('ymax').text = str(int(bbox[3]))

    file_name = os.path.splitext(os.path.basename(file_path))[0] + f'_augmented_{iteration}.xml'
    new_file_path = os.path.join(new_folder, file_name)
    tree.write(new_file_path, encoding='utf-8', xml_declaration=True)

# Function to update JSON annotations
def update_json(file_path, new_folder, bboxes, labels, iteration):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for obj, bbox in zip(data['objects'], bboxes):
        obj['points']['exterior'] = [
            [int(bbox[0]), int(bbox[1])],
            [int(bbox[2]), int(bbox[3])]
        ]

    file_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0] + f'_augmented_{iteration}.jpg.json'
    new_file_path = os.path.join(new_folder, file_name)
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to parse folder and apply augmentations
def parse_folder(folder, ann_folder, output_image_folder, output_ann_folder, augmentations_per_image,k):
    j=0
    files = os.listdir(ann_folder)
    for file in files:
        if j<k:
            ann_ext = os.path.splitext(file)[-1]
            file_path = os.path.join(ann_folder, file)
            if ann_ext == '.xml':
                tree = ET.parse(file_path)
                root = tree.getroot()
                bboxes = []
                labels = []
                for obj in root.findall('object'):
                    class_title = obj.find('name').text
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_title)
                
                image_name = os.path.splitext(file)[0] + '.jpg'
                image_path = os.path.join(folder, image_name)
                for i in range(augmentations_per_image):
                    j+=1
                    new_filename = os.path.join(output_image_folder, os.path.splitext(file)[0] + f'_augmented_{i}.jpg')
                    new_bboxes, new_labels = augment_and_save(image_path, augmentations, new_filename, bboxes, labels)
                    if new_bboxes:
                        update_xml(file_path, output_ann_folder, new_bboxes, new_labels, i)
            elif ann_ext == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                bboxes = []
                labels = []
                for obj in data['objects']:
                    class_title = obj['classTitle']
                    points = obj['points']['exterior']
                    xmin = points[0][0][0]
                    ymin = points[0][0][1]
                    xmax = points[0][1][0]
                    ymax = points[0][1][1]
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_title)
                
                image_name = os.path.splitext(file)[0] 
                image_path = os.path.join(folder, image_name)
                for i in range(augmentations_per_image):
                    j+=1
                    new_filename = os.path.join(output_image_folder, os.path.splitext(os.path.splitext(file)[0])[0] + f'_augmented_{i}.jpg')
                    new_bboxes, new_labels = augment_and_save(image_path, augmentations, new_filename, bboxes, labels)
                    if new_bboxes:
                        update_json(file_path, output_ann_folder, new_bboxes, new_labels, i)


# Define the augmentations
augmentations = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    Affine(shear=(0, 20), p=0.5),
    Blur(blur_limit=7, p=0.5)
], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Directories
image_dir = '/kaggle/working/TrafficLightClassification/yellow_images/images'
ann_dir = '/kaggle/working/TrafficLightClassification/yellow_images/ann'
output_image_dir = '/kaggle/working/TrafficLightClassification/yellow_images/aug_images'
output_ann_dir = '/kaggle/working/TrafficLightClassification/yellow_images/aug_ann'

# Ensure output directories exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_ann_dir, exist_ok=True)

parser = argparse.ArgumentParser(description='Input hyperparameters')
parser.add_argument('--k', type=int, help='Number of augmentated images to be made')
parser.add_argument('--aug', type=int, help='Enter the number of augmentations per image')
args = parser.parse_args()


if __name__ == '__main__':
    
    # Parse folder and apply augmentations
    parse_folder(image_dir, ann_dir, output_image_dir, output_ann_dir, args.aug, args.k)