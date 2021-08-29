import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response, jsonify
from werkzeug.exceptions import BadRequest
import os
import numpy as np

from werkzeug.wrappers import response


# creating flask app
app = Flask(__name__)


# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {}
# create a list of keys to use them in the select part of the html code
listOfKeys = []

for r, d, f in os.walk("models_train"):
    for file in f:
        if ".pt" in file:
            # example: file = "model1.pt"
            # the path of each model: os.path.join(r, file) 
            dictOfModels[os.path.splitext(file)[0]] = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(r, file), force_reload=False) # later set to True
            # you would obtain: dictOfModels = {"model1" : model1 , etc}
    
    for key in dictOfModels :
        listOfKeys.append(key)     # put all the keys in the listOfKeys


# get method
@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("index.html", len = len(listOfKeys), listOfKeys = listOfKeys)


# custom post request
# version 1.0: returns only 1 object per image, can match multiple model outputs for the same object
@app.route('/detections', methods=['POST'])
def return_json():
    file = extract_img(request)
    img_bytes = file.read()
    response = []

    emotion_result = get_prediction(img_bytes,dictOfModels['yolov5s_emotion'])
    age_result = get_prediction(img_bytes,dictOfModels['yolov5s_age'])

    emotion_predition = emotion_result.pred[0].numpy()
    age_predition = age_result.pred[0].numpy()

    print(emotion_predition)
    print(age_predition)

    # getting the baseline predictions
    for id, prediction in enumerate(emotion_predition):
        print("emo prediction", id, "is", prediction)
        result = {
            "object": id,
            "emo_box": list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4])),
            "emo_class" : int(prediction[5]),
            "emo_class_label" : emotion_result.names[int(prediction[5])],
            "emo_confidence" : float("{0:.2f}".format(prediction[4]))
        }
        response.append(result)
    
    # match age model with the baseline predition
    for prediction in age_predition:
        print("age prediction is", prediction)
        box = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        for object in response:
            diff = abs(np.array(object["emo_box"]) - np.array(box))
            isSameObject = (diff < [10]* 4 ).all() # set error threshold
            if isSameObject:
                object["age_class"] = int(prediction[5])
                object["age_class_label"] = age_result.names[int(prediction[5])]
                object["age_confidence"] = float("{0:.2f}".format(prediction[4]))

    return jsonify({"response":response}), 200


# post method
@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()

    # choice of the model
    # results = get_prediction(img_bytes,dictOfModels[request.form.get("model_choice")])
    # print(f'User selected model : {request.form.get("model_choice")}')

    age_result = get_prediction(img_bytes,dictOfModels['yolov5s_age'])
    emotion_result = get_prediction(img_bytes,dictOfModels['yolov5s_emotion'])
    results = emotion_result
    # updates results.imgs with boxes and labels
    results.render()

    # encoding the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response

# dummy post method
@app.route('/dummy', methods=['GET', 'POST'])
def dummy():
    if request.method == 'POST':
        word = request.form["greeting"]
        print(word)
    else:
        word = "No sound"
    return {"status" : "success", "message" : word}

def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
        
    file = request.files['file']
    
    if file.filename == '':
        raise BadRequest("Given file is invalid")
        
    return file


# inference fonction
def get_prediction(img_bytes,model):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=640)  
    return results


def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['left']) + int(res['width'])
        bottom = int(res['top']) + int(res['height'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        print(left, top, right, bottom)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick//3)
    return imageData
    

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
