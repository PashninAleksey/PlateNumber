import os
from PIL import Image
import numpy as np
import cv2 as cv
import easyocr
import torch
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import xml.dom.minidom
import logging
import warnings
import multiprocessing
import socket
from pathlib import Path

def filter_string(input_string):
    input_string = input_string.upper()
    filtered_string = input_string.replace(" ", "")
    filtered_string = list(filtered_string)

    for i in [0, 4, 5]:
        if filtered_string[i] == "4":
            filtered_string[i] = "А"
        if filtered_string[i] == "8":
            filtered_string[i] = "В"
        if filtered_string[i] == "0":
            filtered_string[i] = "О"

    if len(filtered_string) == 7:
        len_number = [1, 2, 3, 6]
    elif len(filtered_string) == 8:
        len_number = [1, 2, 3, 6, 7]
    else:
        len_number = [1, 2, 3, 6, 7, 8]
    for i in len_number:
        if filtered_string[i] == "А":
            filtered_string[i] = "4"
        if filtered_string[i] == "В":
            filtered_string[i] = "8"
        if filtered_string[i] == "О":
            filtered_string[i] = "0"

    filtered_string = "".join(filtered_string)

    return filtered_string


def read_license_plate(box, reader, allowed_characters):
    img_plate = box.copy()
    plate_num = None
    try:
        plate_nums = reader.readtext(
            img_plate,
            allowlist=allowed_characters,
            width_ths=20,
            height_ths=4,
            slope_ths=0.5,
        )
        best_score = 0
        for plate_num_tmp in plate_nums:
            if plate_num_tmp[2] > best_score and len(plate_num_tmp[1]) > 6:
                best_score = plate_num_tmp[2]
                plate_num = plate_num_tmp
        if best_score == 0:
            for plate_num_tmp in plate_nums:
                if plate_num_tmp[2] > best_score:
                    best_score = plate_num_tmp[2]
                    plate_num = plate_num_tmp
    except Exception as ex:
        logging.error(f"Ошибка при распознавании номера: {ex}")
        plate_num = None

    return plate_num


def preprocess_license_plate(
    image,
    treshold2zero,
    target_score,
    allowed_characters,
    reader
):
    
    try:
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
        image = clahe.apply(image)
        image = cv.resize(image, (320, 80), interpolation=cv.INTER_LANCZOS4)
    except Exception as ex:
        logging.error(f"Ошибка при обработке номера: {ex}")

    if treshold2zero:
        best_score = 0
        for i in range(1, 250, 5):
            _, imagebin = cv.threshold(image, i, 255, cv.THRESH_TOZERO)
            plate_num = read_license_plate(imagebin, reader, allowed_characters)
            if plate_num is not None:
                if plate_num[2] > best_score and len(plate_num[1]) > 6:
                    best_score = plate_num[2]
                    plate_num_result = plate_num
                    if best_score > target_score:
                        break
        if plate_num is not None:
            plate_num = plate_num_result

    else:
        plate_num = read_license_plate(image, reader, allowed_characters)

    return plate_num


def read_settings_from_xml(xml_path):
    if not os.path.exists(xml_path):
        create_default_settings(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    settings = {}
    for param in root:
        settings[param.tag] = param.text.strip()

    settings["treshold2zero"] = settings["treshold2zero"].lower() == "true"
    settings["target_score"] = float(settings["target_score"])
    settings["car_detected"] = settings["car_detected"].lower() == "true"

    return settings


def create_default_settings(xml_path="PlateNumberAI.xml"):
    root = ET.Element("settings")
    default_params = {
        "car_path_model": "models/yolov8l.pt",
        "plate_path_model": "models/best8n.pt",
        "easy_ocr_path_model": "easyOcrModel/",
        "name_easy_ocr": "custom_example",
        "path_save_plate": "plate.bmp",
        "path_save_number": "number.txt",
        "treshold2zero": "True",
        "target_score": "0.99",
        "car_detected": "True",
        "allowed_characters": "0123456789АВЕКМНОРСТУХ ",
        "server_address": "127.0.0.1",
        "server_port": "12345"
    }

    for param, default_value in default_params.items():
        elem = ET.SubElement(root, param)
        elem.text = default_value

    tree = ET.ElementTree(root)
    xml_str = ET.tostring(root, encoding="utf-8")
    xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")

    with open(xml_path, "wb") as f:
        f.write(xml_str.encode("utf-8"))

def warm_up(model, reader, car_model=None):
    
    test_image = Image.new('RGB', (100, 100), color = (0, 0, 0))
    model(test_image, verbose=False)
    reader.readtext(np.array(test_image), detail=0)
    if car_model is not None:
        car_model(test_image, verbose=False)

def main():
    # Чтение параметров из XML
    setting_path = "PlateNumberAI.xml"
    settings = read_settings_from_xml(setting_path)

    # Задание путей к моделям и файлам и прочие настройки на основе параметров из XML
    car_path_model = settings["car_path_model"]
    plate_path_model = settings["plate_path_model"]
    easy_ocr_path_model = settings["easy_ocr_path_model"]
    name_easy_ocr = settings["name_easy_ocr"]
    path_save_plate = settings["path_save_plate"]
    path_save_number = settings["path_save_number"]
    treshold2zero = settings["treshold2zero"]
    target_score = settings["target_score"]
    car_detected = settings["car_detected"]
    allowed_characters = settings["allowed_characters"]
    server_address = settings["server_address"]
    server_port = int(settings["server_port"])

    number = None
    vehicles = [2, 3, 5, 7]
    deviceYolo = "cpu"
    deviceEasyOcr = False
    #переменная для проскакивания если что то не так
    cont=True
    response_ok = "Ok"
    response_err = "Error"



    if torch.cuda.is_available():
        deviceYolo = 0
        deviceEasyOcr = True

    # Инициализация модели для обнаружения номеров и их обрезки
    try:
        model_plate = YOLO(plate_path_model)
        reader = easyocr.Reader(
        ["ru"],
        model_storage_directory=os.path.join(easy_ocr_path_model, "model"),
        user_network_directory=os.path.join(easy_ocr_path_model, "user_network"),
        recog_network=name_easy_ocr,
        gpu=deviceEasyOcr,
        )
    except Exception as ex:
        logging.error(f"Ошибка при загрузке модели номера или определителя символов: {ex}")
           
    
    # Настройка логгирования
    logging.basicConfig(
        filename="plate_number.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(module)s - %(levelname)s - %(funcName)s: %(lineno)d - %(message)s",
        datefmt='"%Y-%m-%d %H:%M:%S"',
    )
    
    # Обнаружение транспорта
    if car_detected:
        car_model = YOLO(car_path_model)
         # Прогрев моделей
        warm_up(model_plate, reader, car_model)
    else:
        warm_up(model_plate, reader)
        
    
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((server_address,server_port))
    server.listen(5)
    while True:
        try:
           
            client, addr = server.accept()
            data = client.recv(1024).decode()
            base_file_path = Path(data.strip())
               
            

            if cont:

                # Чтение изображения в виде матрицы
                try:
                    image = cv.imread(base_file_path)
                except Exception as ex:
                    logging.error(f"Ошибка при чтении изображения: {ex}")
                    cont = False
        
                if cont:
                    # Преобразование в изображение для обрезки
                    image_tmp = Image.fromarray(image)

                    # Перевод в черно-белый, если цветной
                    if len(np.array(image).shape) == 3:
                        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                    image = Image.fromarray(image)

                    # Обнаружение транспорта
                    if car_detected:
                        
                        detections = car_model(image, device=deviceYolo, verbose=False)[0]
                        best_score = 0
                        for detection in detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            if int(class_id) in vehicles:
                                if score > best_score:
                                    best_score = score
                                    car = image_tmp.crop((int(x1), int(y1), int(x2), int(y2)))

                        # cv.imwrite("car.jpg", np.array(car))
                        license_plates = model_plate(car, device=deviceYolo, verbose=False)[0]
                        best_score = 0
                        for license_plate in license_plates.boxes.data.tolist():
                            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                            plate = car.crop(
                                (int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2))
                            )
                            plate = np.array(plate)
                            numb_tmp = preprocess_license_plate(
                                plate,
                                treshold2zero,
                                target_score,
                                allowed_characters,
                                reader
                            )
                            if numb_tmp is not None:
                                if len(numb_tmp[1]) > 6:
                                    plate_itog = plate
                                    number = numb_tmp[1]

                    # Только номер
                    else:
                        license_plates = model_plate(image, device=deviceYolo, verbose=False)[0]

                        if len(license_plates.boxes.data.tolist()) > 0:
                            best_score = 0
                            for license_plate in license_plates.boxes.data.tolist():
                                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                                plate = image.crop(
                                    (int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2))
                                )
                                plate = np.array(plate)
                                numb_tmp = preprocess_license_plate(
                                    plate,
                                    treshold2zero,
                                    target_score,
                                    allowed_characters,
                                    reader
                                )
                                if numb_tmp is not None:
                                    if len(numb_tmp[1]) > 6:
                                        plate_itog = plate
                                        number = numb_tmp[1]

                    # Сохранение номера и кадра номера
                    if number is not None:
                        cv.imwrite(
                            os.path.join(os.path.dirname(base_file_path), path_save_plate), plate_itog
                        )
                        #number = filter_string(number)
                        with open(
                            os.path.join(os.path.dirname(base_file_path), path_save_number), "w"
                        ) as file:
                            file.write(number.lower()) 
                
                client.send(response_ok.encode())
                
        except Exception as ex:
            
            client.send(response_err.encode())
            logging.error(f"Произошла ошибка: {ex}")
        finally:
            # Закрываем соединение
            client.close()
      
            


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Игнорировать все предупреждения
    warnings.filterwarnings("ignore")
    # Вызов основной функции
    main()
