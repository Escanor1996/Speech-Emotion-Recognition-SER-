
import torch
from commons import get_model,get_tensor


class_names=['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
model=get_model()
def prediction(image_bytes):
    tensor=get_tensor(image_bytes)
    outputs=model(tensor)
    _,prediction=outputs.max(1)
    category=prediction.item()
    emotion=class_names[category]

    return emotion