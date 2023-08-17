from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io


app = FastAPI()
# yolo_classes = ["fire"]
yolo_classes = [
    "fire1","fire2","fire3","fire4"
]
model = ort.InferenceSession('model details/weights/best.onnx')

# def process_image(image_data):
#     
#     img = Image.open(io.BytesIO(image_data))
#     input, img_width, img_height = preprocess_image(img)
#     output = run_model(input)
#     return postprocess_output(output, img_width, img_height)

def process_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    input_data, img_width, img_height = preprocess_image(img)
    output = run_model(input_data)
    return postprocess_output(output, img_width, img_height)

# Preprocessing function
# def preprocess_image(image):
#     img = Image.open(image)
#     img_width, img_height = img.size
#     img = img.resize((640, 640))
#     img = img.convert("RGB")
#     input = np.array(img) / 255.0
#     input = input.transpose(2, 0, 1)
#     input = input.reshape(1, 3, 640, 640)
#     return input.astype(np.float32), img_width, img_height
def preprocess_image(image):
    img_width, img_height = image.size
    img = image.resize((640, 640))
    img = img.convert("RGB")
    input_data = np.array(img) / 255.0
    input_data = input_data.transpose(2, 0, 1)
    input_data = input_data.reshape(1, 3, 640, 640)
    return input_data.astype(np.float32), img_width, img_height

# def run_model(input): 
#     outputs = model.run(["output0"], {"images":input})
#     return outputs[0]
def run_model(input_data): 
    outputs = model.run(["output0"], {"images": input_data})
    return outputs[0]

# Post-processing function
def postprocess_output(output, img_width, img_height):
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.25:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.5]

    return result

def iou(box1,box2):
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):

    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):

    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)



@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    boxes = process_image(image)


    structured_boxes = []
    for box in boxes:
        structured_box = {
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
            "label": box[4],
            "confidence": box[5]
        }
        structured_boxes.append(structured_box)

    return structured_boxes