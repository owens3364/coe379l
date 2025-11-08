from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

MODEL_PATH = 'modified_lenet7.h5'
model = tf.keras.models.load_model(MODEL_PATH)
input_shape = model.input_shape[1:3]
class_names = ['damage', 'no_damage']

app = FastAPI(
  title='Damage Classification Model API',
  description='Simple inference server for the Modified LeNet model 7',
  version='0.0.1'
)

@app.get('/summary')
def get_summary():
  try:
    summary_dict = {
      'model_name': 'Modified LeNet 7',
      'input_shape': input_shape,
      'num_classes': len(class_names),
      'class_names': class_names
    }
    return JSONResponse(content=summary_dict)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

def preprocess_image(image_bytes: bytes):
  img = Image.open(BytesIO(image_bytes)).convert('RGB')
  img = img.resize(input_shape)
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

@app.post('/inference')
async def inference(file: UploadFile = File(...)):
  try:
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)
    pred = model.predict(img_array)
    pred_label = 'no_damage' if pred[0][0] > 0.5 else 'damage'

    return JSONResponse(content={'prediction': pred_label})
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

# uvicorn app:app --host 0.0.0.0 --port 8000