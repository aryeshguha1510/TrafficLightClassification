
import json
from PIL import Image,ImageDraw
import os
import copy
def parse_folder(images_folder_name,annotations_folder_name):
    files=os.listdir(images_folder_name)
    ann_images=os.listdir(annotations_folder_name)
    for file in ann_images:
        file=os.path.join(annotations_folder_name,file)
        data=json_read(file)
        result=extract_classTitles(data)
        if result['objects']==None or result['objects']==[]:
            continue
        image_name=os.path.splitext(os.path.basename(file))[0]
        json_path=os.path.join('D:/dataset_clean/test_labels',os.path.basename(file))
        json_write(result,json_path)
        image_path=os.path.join(images_folder_name,image_name)
        train_image=Image.open(image_path)
        save_image(image_name,'D:/dataset_clean/test',train_image)
def json_read(json_file_path):
    with open(json_file_path,'r') as file:
        data=json.load(file)
        return data
def json_write(result,json_file_new_path):
    with open(json_file_new_path,'w+') as file:
        json.dump(result,file,indent=4)
def extract_classTitles(data):
    copy_data=copy.deepcopy(data)
    data['objects']=[]
    for obj in copy_data['objects']:
        if obj['classTitle']=="go traffic light" or obj['classTitle']=="stop traffic light" or obj['classTitle']=="warning traffic light":
            data['objects'].append(obj)
    return data
# json_file_path='C:/Users/HP PAVILION/Desktop/lisa-traffic-light-DatasetNinja/train/ann/dayClip1--00000.jpg.json'

# def draw_bounding_boxes(image_path,annotations):
#     image= Image.open(image_path)
#     draw=ImageDraw.Draw(image)
#     for annotation in annotations:
#         class_title=annotation['classTitle']
#         points=annotation['points']['exterior']
#         top_left=tuple(points[0])
#         bottom_right=tuple(points[1])
#         draw.rectangle([top_left,bottom_right],outline="red",width=2)
#         # draw.text(top_left,class_title,fill="green")
#     return image
def save_image(image_name,folder_name,image):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    output_path=os.path.join(folder_name,image_name)
    image.save(output_path)
# image_path="C:/Users/HP PAVILION/Desktop/lisa-traffic-light-DatasetNinja/train/img/dayClip1--00000.jpg"
# image=draw_bounding_boxes(image_path,result)
# save_image('dayClip1--00000.jpg','C:/Users/HP PAVILION/Desktop/dataset_clean/train_labels',image)
parse_folder('C:/Users/HP PAVILION/Desktop/lisa-traffic-light-DatasetNinja/test/img','C:/Users/HP PAVILION/Desktop/lisa-traffic-light-DatasetNinja/test/ann')