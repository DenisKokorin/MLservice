import io
import os
from PIL import Image
from ML.load import load_model

model = None

def load_ml():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

def save_image(image: bytes, file_name: str):
    try:
        img = Image.open(io.BytesIO(image))
    except Exception as e:
        print(f"Ошибка при открытии изображения: {e}")
        return None
    
    try:
        if not os.path.exists("savedIMG"):
            os.makedirs("savedIMG")
        path = f'savedIMG/{file_name}'
        img.save(path)
        return path
    except Exception as e:
        print(f"Ошибка при сохранении изображения: {e}")
        return None

def get_result_from_ml(path):
    if model is None:
        print("Модель не загружена. Пожалуйста, загрузите модель перед выполнением предсказаний.")
        return None
    
    try:
        results = model.predict(source=path, conf=0.5)
        res = {}
        for r in results:
            boxes = r.boxes.data.tolist()
            for box in boxes:
                res["xmin"] = box[0]
                res["ymin"] = box[1]
                res["xmax"] = box[2]
                res["ymax"] = box[3]
                res["confidence"] = box[4]
                res["class"] = box[5]
                res["class_name"] = model.names[int(box[5])]
        return res
    except Exception as e:
        print(f"Ошибка при получении результатов из модели: {e}")
        return None
