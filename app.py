import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response, jsonify
from werkzeug.exceptions import BadRequest
import os
import numpy as np
from werkzeug.wrappers import response
import json

# create flask app
app = Flask(__name__)
json_dir = "jsons/"

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}
# create a list of keys to use them in the select part of the html code
listOfKeys = []

for r, d, f in os.walk("models_train"):
    for file in f:
        if ".pt" in file:
            # the path of each model: os.path.join(r, file) 
            dictOfModels[os.path.splitext(file)[0]] = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(r, file), force_reload=True) 
    
    for key in dictOfModels :
        listOfKeys.append(key)     # put all the keys in the listOfKeys

# post method
@app.route('/', methods=['POST'])
def predict():
    """
    1. Perform inference for all four models
    2. Filter duplicate objects
    3. Save JSON file 
    4. Re-draw bounding boxes
    5. Return image
    """
    file = extract_img(request)
    img_bytes = file.read() # 'bytes' object
    UUID = file.filename[:file.filename.find(".")]
    
    # Inference and filter
    results = get_prediction(img_bytes,dictOfModels['yolov5l_emotion'])
    preditions = results.pred[0].numpy()
    preditions = get_distinct_objects(preditions)

    # Generate and save JSON text
    response = []

    for id, prediction in enumerate(preditions):
        result = {
            "object": id+1,
            "emo_box": list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4])),
            "emo_class" : int(prediction[5]),
            "emo_class_label" : results.names[int(prediction[5])].capitalize(),
            "emo_confidence" : float("{0:.2f}".format(prediction[4])),
            "age_class" : -1, 
            "age_class_label" : "Unknown",  
            "age_confidence" : 0.0,
            "ethnicity_class" : -1,
            "ethnicity_class_label" : "Unknown",  
            "ethnicity_confidence" : 0.0,
            "gender_class" : -1,
            "gender_class_label" : "Unknown",  
            "gender_confidence" : 0.0
        }
        response.append(result)

    match_objects(img_bytes, response, "age")
    match_objects(img_bytes, response, "ethnicity")
    match_objects(img_bytes, response, "gender")

    with open(os.path.join(json_dir, UUID + ".json"), 'w', encoding="utf-8") as f:
        json.dump(response, f)

    # Re-draw bounding box and return image
    objects = []
    for id, prediction in enumerate(preditions):
        coordinates = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        object = {
            "left" : coordinates[0],
            "top" : coordinates[1],
            "right" : coordinates[2],
            "bottom" : coordinates[3],
            "label" : "Person " + str(id+1)
        }
        objects.append(object)

    img = Image.open(io.BytesIO(img_bytes))
    img_draw = drawBoundingBoxes(np.array(img), objects)
    RGB_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    im_arr = cv2.imencode('.jpg', RGB_img)[1]
    response = make_response(im_arr.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    return response

@app.route('/detections', methods=['POST'])
def return_json():
    """
    Return the json format result.
    POST Request with raw text (name of the image)
    """
    json_name = request.data.decode("utf-8")

    with open(os.path.join(json_dir, json_name + ".json"), 'r', encoding="utf-8") as f:
        response = json.load(f)

    for i in os.listdir(json_dir):
        file_path = os.path.join(json_dir, i)
        os.remove(file_path)

    return jsonify({"response":response}), 200

def match_objects(img_bytes, response, output):
    """
    Put the detection result in the corresponding position is the json.
    - Extract the class id, class label and confidence.
    - Duplicate object detections is allow here and will be filtered out later.
    """
    results = get_prediction(img_bytes,dictOfModels['yolov5l_'+output])
    predictions = results.pred[0].numpy()
    for prediction in predictions:
        box = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        for object in response:
            diff = abs(np.array(object["emo_box"]) - np.array(box))
            isSameObject = (diff < [20]* 4 ).all() # set error threshold
            if isSameObject:
                object[output+"_class"] = int(prediction[5])
                if output == "ethnicity":
                    object[output+"_class_label"] = results.names[int(prediction[5])].capitalize().split("-")[0]
                else:
                    object[output+"_class_label"] = results.names[int(prediction[5])].capitalize()
                object[output+"_confidence"] = float("{0:.2f}".format(prediction[4]))
    

def get_distinct_objects(predictions):
    """
    Filter out lower confidence bounding boxes output.
    predictions: return only the highest confident detected object for each location.
    """
    distinct_index = set(list(range(len(predictions))))

    for i in range(len(predictions)):
        box_1 = list(map(lambda x: float("{0:.2f}".format(x)), predictions[i][:4]))
        for j in range(len(predictions)):
            box_2 = list(map(lambda x: float("{0:.2f}".format(x)), predictions[j][:4]))
            diff = abs(np.array(box_1) - np.array(box_2))
            if (diff < [20]* 4 ).all():
                conf_1 = float("{0:.2f}".format(predictions[i][4]))
                conf_2 = float("{0:.2f}".format(predictions[j][4]))
                if i!=j:
                    if conf_1>=conf_2:
                        distinct_index.discard(j)
                    else:
                        distinct_index.discard(i)
    predictions = list(np.array(predictions)[list(distinct_index)])
    return predictions

def extract_img(request):
    """
    Checking if image uploaded is valid
    """
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
        
    file = request.files['file']
    
    if file.filename == '':
        raise BadRequest("Given file is invalid")
        
    return file


def get_prediction(img_bytes,model):
    """
    Inference function using the selected model
    """
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=640)  
    return results

def drawBoundingBoxes(imageData, inferenceResults): 
    """
    re-drawing the bounding boxes in order to annotate the id of the object.

    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right']) 
        bottom = int(res['bottom']) 
        label = res['label']
        color = (255, 199, 50)
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 300)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 5), 2, 1e-3 * (imgHeight + imgWidth), color, thick//2)
    return imageData
    

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0', port=5000, use_reloader=False)
