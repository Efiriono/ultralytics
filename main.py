import cv2
from ultralytics import YOLO
import random
import os
import logging
import json

list_point_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
point_pairs = [
    (5, 7), (7, 9), (11, 13), (13, 15),  # Левая и правая руки
    (5, 11), (6, 12),  # Торс
    (11, 12),  # Между ног
    (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
    (5, 6),  # Соединение плечей
    (5, 7), (7, 9),  # Левая рука
    (6, 8), (8, 10),  # Правая рука
    (11, 13), (13, 15),  # Нога левая
    (12, 14), (14, 16)  # Нога правая
]

def process_video_with_tracking(model, input_video_path, output_dir="results", show_video=False, save_video=True):
    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Создание выходной директории
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_video_path = os.path.join(output_dir, "output_video.mp4")
    json_file_path = os.path.join(output_dir, "coordinates.json")

    # Открытие входного видеофайла
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Получение параметров входного видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Определение параметров для записи выходного видео
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, iou=0.6, conf=0.5, persist=True, imgsz=608, verbose=False,
                              tracker="botsort.yaml")

        if results[0].boxes.id is not None:  # Проверяем, что id не None -> существуют треки
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints_all = results[0].keypoints.xy.cpu().numpy()

            for person_index, person_id in enumerate(ids):
                person_id = int(person_id)  # Преобразуем numpy.int32 в стандартный int
                keypoints = keypoints_all[person_index]
                if person_id not in data:
                    data[person_id] = {}
                data[person_id][frame_number] = {list_point_names[i]: (int(x), int(y)) for i, (x, y) in enumerate(keypoints)}

                # Генерация случайного цвета для объекта на основе его ID
                random.seed(person_id)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Рисуем точки
                for j, point in enumerate(keypoints):
                    x, y = point
                    if x > 0 and y > 0:  # Проверяем, что обе координаты больше нуля
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                # Соединяем точки линиями
                for pair in point_pairs:
                    start, end = pair
                    if keypoints[start][0] > 0 and keypoints[start][1] > 0 and keypoints[end][0] > 0 and \
                            keypoints[end][1] > 0:  # Проверяем, что обе пары координат больше нуля
                        x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                        x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # Логирование номера текущего кадра
        logging.info(f"Processing frame {frame_number}")

        if save_video:
            out.write(frame)

        if show_video:
            cv2.imshow("frame", frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

        frame_number += 1

    # Сохранение координат в JSON файл
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    # Освобождение ресурсов
    cap.release()
    if save_video:
        out.release()

    cv2.destroyAllWindows()

# Загрузка модели
model = YOLO('yolov8m-pose.pt')

# Принудительное использование CPU
model.to('cpu')

# Запуск обработки видео
process_video_with_tracking(model, "test_video.mp4", output_dir="results", show_video=False, save_video=True)

print("Обработка завершена. Результаты сохранены в папке 'results'.")
