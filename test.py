import cv2
from ultralytics import YOLO

# Инициализация модели YOLO
model = YOLO('yolov8n.pt')

# Открытие видео
video_path = 'test_video_solo_mhw.mp4'
cap = cv2.VideoCapture(video_path)

# Параметры для записи видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_number = 0
tracking_point = None  # Координаты точки для отслеживания
output_file = 'tracked_coordinates.txt'

with open(output_file, 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Выполнение предсказания
        results = model(frame, device='cpu')

        # Проверка наличия обнаруженных объектов
        if results and len(results[0].boxes) > 0:
            # Получение координат первого обнаруженного человека
            box = results[0].boxes[0].xyxy[0]  # Используем первую детекцию (нулевой индекс)
            x1, y1, x2, y2 = box[:4]
            shoulder_y = int(y1 + (y2 - y1) * 0.25)  # Верхняя четверть bounding box-а
            tracking_point = (int((x1 + x2) / 2), shoulder_y)  # Центр по горизонтали и верхняя четверть по вертикали

            # Отображение точки на видео
            cv2.circle(frame, tracking_point, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({tracking_point[0]}, {tracking_point[1]})",
                        (tracking_point[0] + 10, tracking_point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Запись координат в файл
            f.write(f"{frame_number}, {tracking_point[0]}, {tracking_point[1]}\n")

        # Запись обработанного кадра в видеофайл
        out.write(frame)
        frame_number += 1

print("Обработка завершена")