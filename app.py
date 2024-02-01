"""
Build a PyTorch model that can be used for prediction served out via FastAPI
"""

import io
import json
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import fastapi
from fastapi import File, UploadFile, Request
import uvicorn
import torch
from fastapi.responses import RedirectResponse, HTMLResponse
import base64
#import cv2
import numpy as np

#from prediction_utils import get_args_parser, ddd

app = fastapi.FastAPI()

model = torch.load("model_37_GOODONE.pth", map_location=torch.device('cpu'))

model.eval()
model.cuda() # send weights to gpu, may not work...

model_class_index = json.load(open("class_index.json", encoding="utf-8"))

# def transform_image(image_bytes):
#     my_transforms = transforms.Compose(
#         [
#             transforms.Resize(256),
#             transforms.CenterCrop(256),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    
    #tensor = transform_image(image_bytes=image_bytes)
    
    image = Image.open(io.BytesIO(image_bytes))
    tensor = image.unsqueeze(0)
    outputs = model(tensor)
    _, y_hat = torch.max(outputs.data,1)
    model_pred_idx = str(y_hat.item())
    label_pred = model_class_index[model_pred_idx]
    return model_pred_idx, label_pred

def get_inference(image_bytes):
    
    #tensor = transform_image(image_bytes=image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    #tensor = image.unsqueeze(0)
    #img_c = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #img_t = img_c.transpose([2,0,1])
    img_t = image.transpose([2,0,1])

    img_t = np.expand_dims(img_t, axis=0)
    img_t = img_t/255.0
    img_t = torch.FloatTensor(img_t)

    print('shape of img_t ', np.shape(img_t))
    device = torch.device("cuda")
    img_t = img_t.to(device)
    detections = model(img_t)[0]

    #loop over the detections
    prediction_labels = []
    prediction_vals = []

    preds = []

    #inputname = input_image_path.split('\\')
    
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]

        #print('confidence: ', confidence)

        con = 0.8#0.5#0.8
        COLORS = np.random.uniform(150, 255, size=(len(CLASSES), 3))
        CLASSES = [
            'scratch', 
            'dent', 
            'paint', 
            'pit', 
            'none'
                ]

        inputname = 'TEST!'
        if confidence > con:

            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            #print('prediction val: ', idx-1)
            prediction_vals.append(idx-1)
            label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
            #print("[PREDICTION:] {}".format(label))
            prediction_labels.append(label)
            # draw the bounding box and label on the image
            # cv2.rectangle(img_c, (startX, startY), (endX, endY),
            #     COLORS[idx-1], 2)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.putText(img_c, label, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx-1], 5)
            
            preds.append([inputname[-1],startX, startY, endX, endY, CLASSES[idx-1], float(confidence * 100)])
            
    print(' [PREDICTIONS: ]', prediction_labels, prediction_vals)
    return 'sample_model_pred', 'glass'



@app.get("/")
def index():
    return {"message": "Hello TSI-ML-OPS-Feb-2-2024"}

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

# @app.post("/predictOLD")
# async def predictFIle(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     print('length of inc image', len(image_bytes))
#     model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=image_bytes)
#     return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

@app.post("/make_inference")
async def predictFIle(file: UploadFile = File(...)):
    image_bytes = await file.read()
    print('length of inc image', len(image_bytes))
    model_pred_idx, label_pred = get_inference(image_bytes=image_bytes)
    return {"defect": model_pred_idx, "defectclass": label_pred}


# adding Matthias's route, using requests
# https://fastapi.tiangolo.com/advanced/using-request-directly/
@app.post("/predict")
async def predictRequest(request: Request):
    
  if request.method == 'POST':
    content_type = request.headers.get('Content-type')
        
    if (content_type == 'application/json'):
      
      data = await request.json()
      if not data:
        return
      
      img_string = data.get('file')
      #Clean string
      img_string = img_string[img_string.find(",")+1:]
      img_bytes = base64.b64decode(img_string)      
    elif (content_type == 'multipart/form-data'):
      print('you have multiformish dater!')
      if 'file' not in request.files:
        return {"oops":"no data in form"}
      file = request.files.get('file')
      if not file:
        return
      img_bytes = file.read()
    else: 
      return "Content type is not supported."
  
    if len(img_bytes) > 0:   # not sure if that works like that in Python...
      model_3_pred_idx, label_pred_3, model_10_pred_idx, label_pred_10 = get_prediction(image_bytes=img_bytes)        
      return {"earlyorlateID": model_3_pred_idx, "class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}
    else: 
      return "Cannot extract image data from request"
{"class_name_3": label_pred_3, "diseaseID": model_10_pred_idx, "class_name_10":label_pred_10}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    #app.run(debug=True, host='0.0.0.0', port=8080)