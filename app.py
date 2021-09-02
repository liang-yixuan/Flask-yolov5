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
# version 2.0: returns multiple objects per image, can match multiple model outputs for the same object
@app.route('/detections', methods=['POST'])
def return_json():
    file = extract_img(request)
    img_bytes = file.read()
    response = []

    emotion_result = get_prediction(img_bytes,dictOfModels['yolov5l_emotion'])
    age_result = get_prediction(img_bytes,dictOfModels['yolov5l_age'])
    ethnicity_result = get_prediction(img_bytes,dictOfModels['yolov5l_ethnicity']) ## placeholder for ethnicity model
    gender_result = get_prediction(img_bytes,dictOfModels['yolov5l_gender']) ## placeholder for gender model

    emotion_predition = emotion_result.pred[0].numpy()
    age_predition = age_result.pred[0].numpy()
    ethnicity_predition = ethnicity_result.pred[0].numpy()
    gender_predition = gender_result.pred[0].numpy()

    print("emotion_predition", emotion_predition)
    print("age_predition", age_predition)
    print("ethnicity_predition", ethnicity_predition)
    print("gender_predition", gender_predition)



    # getting the baseline predictions
    for id, prediction in enumerate(emotion_predition):
        result = {
            "object": id+1,
            "emo_box": list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4])),
            "emo_class" : int(prediction[5]),
            "emo_class_label" : emotion_result.names[int(prediction[5])].capitalize(),
            "emo_confidence" : float("{0:.2f}".format(prediction[4])),
            "age_class" : -1, 
            "age_class_label" : "Could not detect",  
            "age_confidence" : 0.0,
            "ethnicity_class" : -1,
            "ethnicity_class_label" : "Could not detect",  
            "ethnicity_confidence" : 0.0,
            "gender_class" : -1,
            "gender_class_label" : "Could not detect",  
            "gender_confidence" : 0.0
        }
        response.append(result)
    
    # match age model with the baseline predition
    for prediction in age_predition:
        box = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        for object in response:
            diff = abs(np.array(object["emo_box"]) - np.array(box))
            isSameObject = (diff < [20]* 4 ).all() # set error threshold
            if isSameObject:
                object["age_class"] = int(prediction[5])
                object["age_class_label"] = age_result.names[int(prediction[5])]
                object["age_confidence"] = float("{0:.2f}".format(prediction[4]))

    # match age model with the baseline predition
    for prediction in ethnicity_predition:
        box = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        for object in response:
            diff = abs(np.array(object["emo_box"]) - np.array(box))
            isSameObject = (diff < [20]* 4 ).all() # set error threshold
            if isSameObject:
                object["ethnicity_class"] = int(prediction[5])
                object["ethnicity_class_label"] = ethnicity_result.names[int(prediction[5])].capitalize()
                object["ethnicity_confidence"] = float("{0:.2f}".format(prediction[4]))

    # match age model with the baseline predition
    for prediction in gender_predition:
        box = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        for object in response:
            diff = abs(np.array(object["emo_box"]) - np.array(box))
            isSameObject = (diff < [20]* 4 ).all() # set error threshold
            if isSameObject:
                object["gender_class"] = int(prediction[5])
                object["gender_class_label"] = gender_result.names[int(prediction[5])].capitalize()
                object["gender_confidence"] = float("{0:.2f}".format(prediction[4]))

    return jsonify({"response":response}), 200


# post method
@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read() # 'bytes' object
    


    # age_result = get_prediction(img_bytes,dictOfModels['yolov5s_age'])
    emotion_result = get_prediction(img_bytes,dictOfModels['yolov5l_emotion'])
    result = emotion_result
    # updates results.imgs with boxes and labels
    # results.render()
    predition = result.pred[0].numpy()
    objects = []

    # getting the baseline predictions
    for id, prediction in enumerate(predition):
        coordinates = list(map(lambda x: float("{0:.2f}".format(x)), prediction[:4]))
        object = {
            "left" : coordinates[0],
            "top" : coordinates[1],
            "right" : coordinates[2],
            "bottom" : coordinates[3],
            "label" : "Person " + str(id+1)
        }
        # print("object", object)
        objects.append(object)


    # # encoding the resulting image and return it
    # for img in results.imgs: # 'numpy.ndarray' object
    
    img = Image.open(io.BytesIO(img_bytes))
    img_draw = drawBoundingBoxes(np.array(img), objects)
    RGB_img = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    im_arr = cv2.imencode('.jpg', RGB_img)[1]
    response = make_response(im_arr.tobytes())
    response.headers['Content-Type'] = 'image/jpeg'
    

    return response


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


def drawBoundingBoxes(imageData, inferenceResults): 
    """Draw bounding boxes on an image.
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
    app.run(debug=True,host='0.0.0.0')
