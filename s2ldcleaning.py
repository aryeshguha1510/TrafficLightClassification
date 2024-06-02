import os
import copy
import xml.etree.ElementTree as ET
from PIL import Image

def parse_folder(images_folder_name, annotations_folder_name):
    files = os.listdir(images_folder_name) 
    ann_images = os.listdir(annotations_folder_name)
    for file in ann_images:
        file = os.path.join(annotations_folder_name, file)
        data = xml_read(file)
        result = extract_classTitles(data)
        if not result['objects']:
            continue
        image_name = os.path.splitext(os.path.basename(file))[0] + ".jpg"  # Assuming the images are in jpg format
        xml_path = os.path.join('D:/YEAR3/MRM/datasets/S2TLD/clean(3)_annots', os.path.basename(file))
        xml_write(result, xml_path)
        image_path = os.path.join(images_folder_name, image_name)
        train_image = Image.open(image_path)
        save_image(image_name, 'D:/YEAR3/MRM/datasets/S2TLD/clean(3)', train_image)

def xml_read(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data = {'objects': []}
    for obj in root.findall('object'):
        class_title = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        data['objects'].append({
            'classTitle': class_title,
            'points': {
                'exterior': [[xmin, ymin], [xmax, ymax]]
            }
        })
    return data

def xml_write(result, xml_file_new_path):
    root = ET.Element('annotation')
    for obj in result['objects']:
        object_elem = ET.SubElement(root, 'object')
        name = ET.SubElement(object_elem, 'name')
        name.text = obj['classTitle']
        bndbox = ET.SubElement(object_elem, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        ymin = ET.SubElement(bndbox, 'ymin')
        xmax = ET.SubElement(bndbox, 'xmax')
        ymax = ET.SubElement(bndbox, 'ymax')
        xmin.text = str(obj['points']['exterior'][0][0])
        ymin.text = str(obj['points']['exterior'][0][1])
        xmax.text = str(obj['points']['exterior'][1][0])
        ymax.text = str(obj['points']['exterior'][1][1])
    tree = ET.ElementTree(root)
    tree.write(xml_file_new_path)

def extract_classTitles(data):
    filtered_data = {'objects': []}
    for obj in data['objects']:
        if obj['classTitle'] in ["green", "red", "yellow"]:
            filtered_data['objects'].append(obj)
    return filtered_data

def save_image(image_name, folder_name, image):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_path = os.path.join(folder_name, image_name)
    image.save(output_path)

parse_folder('D:/YEAR3/MRM/datasets/S2TLD/S2TLD（720x1280）/normal_2/JPEGImages','D:/YEAR3/MRM/datasets/S2TLD/S2TLD（720x1280）/normal_2/Annotations')
