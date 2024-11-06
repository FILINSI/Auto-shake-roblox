import torch
import numpy as np
import cv2
import mss
import pyautogui
import time

# Загрузка обученной модели
model = torch.hub.load('./yolov5', 'custom', path='best.pt', source='local')

# Настройка модели (если необходимо)
model.conf = 0.25  # Уровень уверенности (можно настроить в зависимости от ваших нужд)

# Использование mss для захвата экрана
with mss.mss() as sct:
    # Установить регион для захвата экрана для конкретного монитора
    monitor = sct.monitors[2]  # Если вам нужен второй монитор, используйте индекс `2`

    while True:
        # Захват экрана
        screen = sct.grab(monitor)

        # Преобразование захваченного изображения в формат, который может обработать YOLOv5 (NumPy)
        img_np = np.array(screen)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)  # Преобразование в RGB

        # Масштабирование изображения (для увеличения FPS можно уменьшить размер изображения)
        img_resized = cv2.resize(img_rgb, (420, 420))

        # Передача изображения в модель для предсказания
        results = model(img_resized)

        # Извлечение результатов в DataFrame
        df = results.pandas().xyxy[0]  # Получение результатов в формате pandas DataFrame

        # Если кнопка обнаружена, кликнуть по ней
        for i in range(len(df)):
            if df.loc[i, 'name'] == 'button':  # Убедитесь, что найден именно объект с именем "button"
                # Преобразование координат для исходного изображения
                scale_x = img_np.shape[1] / img_resized.shape[1]
                scale_y = img_np.shape[0] / img_resized.shape[0]

                xmin, ymin, xmax, ymax = df.loc[i, ['xmin', 'ymin', 'xmax', 'ymax']]
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)

                # Вычисление центра кнопки
                x_center = int((xmin + xmax) / 2)
                y_center = int((ymin + ymax) / 2)

                # Выполнить клик с использованием pyautogui
                pyautogui.moveTo(x_center + monitor["left"], y_center + monitor["top"])
                pyautogui.click()

                

        # Отображение того, что видит нейросеть (для отладки)
        for i in range(len(df)):
            xmin, ymin, xmax, ymax = df.loc[i, ['xmin', 'ymin', 'xmax', 'ymax']]
            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)

            # Рисование рамок на изображении
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Detection', img_np)

        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Закрытие окон и освобождение ресурсов
cv2.destroyAllWindows()
