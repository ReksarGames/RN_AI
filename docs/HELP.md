# Help (ZTXAI)

Этот файл описывает меню и параметры GUI. Названия могут отличаться в зависимости от локализации.

## 1) System / Система
- UI Language - язык интерфейса.
- UI Width Scale - масштаб ширины окна.
- Font Scale - масштаб шрифта.
- Print FPS - вывод FPS в консоль.
- Show Motion Speed - отображение скорости движения.
- Show Curve - отображение кривой скорости.
- Show Infer Time - вывод времени инференса.
- Screenshot Separation - разнос потоков захвата (многопоточная схема).

Small Target Enhancement
- Enable Small Target Enhancement - включает усиление мелких целей.
- Enable Small Target Smoothing - сглаживание для мелких целей.
- Adaptive NMS - адаптивный NMS для мелких целей.
- Small Target Boost - дополнительный вес мелких целей.
- Smooth History Frames - сколько кадров использовать для сглаживания.
- Small Target Threshold - порог размера, ниже которого цель считается мелкой.
- Medium Target Threshold - порог для средних целей (для сглаживания).

## 2) Driver / Драйвер
- Move Method - метод передачи движения (например, makcu).
- COM - COM-порт устройства.
- Mask Left/Right/Middle/Side1/Side2 - маскирование соответствующих кнопок.
- Mask X/Y Axis - блокировка осей движения.
- Aim Mask X/Y - маскирование движения только в режиме наведения.
- Mask Wheel - блокировка колесика.

## 3) Bypass / Обход
- Опции обхода зависят от сборки. Обычно маскируют реальные вводы, когда активен драйвер.

## 4) Strafe / Стрейф
- Содержит профили отдачи и смещения для оружия.
- Mouse_re Trajectory - воспроизведение траектории отдачи из файлов.
- Replay Speed - скорость воспроизведения траектории.

## 5) Config / Конфиг
- Save Config - сохранить текущие настройки в `cfg.json`.
- Профили: добавление/удаление групп, выбор активной группы.

## 6) Aim / Наведение
Aim Controller
- PID - классический контроллер.
- Sunone - расширенный контроллер (с Kalman/Prediction).

PID Controller Parameters
- X/Y Proportional (Kp) - усиление по оси X/Y.
- X/Y Integral (Ki) - накопление ошибки.
- X/Y Derivative (Kd) - подавление резких изменений.
- X/Y Limit - ограничение интегральной части.
- X/Y Smooth - сглаживание выходного сигнала.
- Smooth Algorithm - сила алгоритма сглаживания.
- Smooth Deadzone - зона, где сглаживание не применяется.
- Move Deadzone - минимальный порог движения.

Sunone Settings
- Enable Smoothing - включает сглаживание.
- Tracking Smooth - степень сглаживания трекинга.
- Smoothness - общая мягкость движения.

Kalman (позиция)
- Kalman Process Noise (Q) - насколько фильтр "верит" в движение. Больше Q = быстрее реакция, но больше дрожания.
- Kalman Measurement Noise (R) - насколько фильтр "верит" в измерения. Больше R = сильнее сглаживание, но больше задержка.
- Kalman Speed X/Y - масштаб скорости по осям.
- Reset Threshold - порог сброса при резком скачке.

Prediction (предсказание)
- Prediction Mode:
  - Standard - прогноз по скорости без Kalman.
  - Kalman - скорость берется из Kalman.
  - Kalman + Standard - комбинированный режим.
- Prediction Interval - интервал обновления прогноза.
- Kalman Lead (ms) - опережение по времени.
- Kalman Max Lead (ms) - ограничение максимального опережения.
- Velocity Smoothing - сглаживание скорости перед прогнозом.
- Velocity Scale - масштаб скорости.
- Prediction Kalman Q/R - отдельные Q/R для Kalman в prediction.

Speed Curve
- Min Speed / Max Speed - минимальная/максимальная скорость движения.
- Snap Radius - радиус "прилипания".
- Near Radius - радиус, где скорость уменьшается.
- Curve Exponent - степень кривой.
- Snap Boost - усиление в зоне snap.

Trigger
- Auto Trigger - автонажатие при наведении.
- Continuous Trigger - удержание.
- Trigger Delay - задержка перед нажатием.
- Press Duration - длительность нажатия.
- Trigger Cooldown - задержка между срабатываниями.
- Random Delay - случайная задержка.
- X/Y Trigger Scope - размер зоны триггера.
- X/Y Trigger Offset - смещение зоны.
- Trigger Only - без наведения, только триггер.
- Trigger Recoil - отдача при триггере.

Aim Position
- Aim Position - позиция внутри бокса цели (0.0 = верх, 1.0 = низ).
- Aim Position 2 - вторая точка для двухэтапного наведения.

Smart Target / Dynamic Scope
- Smart Target Lock - удержание текущей цели для стабильности.
- Min Scope - минимальный размер области.
- Shrink/Recover Duration - скорость уменьшения/восстановления области.

Aim Weights
- Distance Weight - приоритет ближних целей.
- Center Weight - приоритет целей ближе к центру.
- Size Weight - приоритет крупных целей.

## 7) Classes / Классы
- Inference Classes - список классов, доступных в модели.
- Class Names File - загрузка имен из файла.
- Class Names (Manual) - ручной ввод имен (по строке на класс).
- Apply Names - применить введенный список.
- Class Priority - порядок приоритета (например `0-1-2-3`).
- Class Aim Config - выбор класса и индивидуальная настройка параметров.
- Confidence Threshold - минимальная уверенность детекции для класса.
- IOU - порог объединения боксов.

## 8) Capture / Захват
- Capture Source: Standard / OBS / Capture Card.
- OBS IP / Port / FPS - параметры источника OBS.
- Capture Device / FPS / Resolution / Crop - параметры карты захвата.
- Capture Offset X/Y - смещение области захвата от центра.

## 9) Debug / Отладка
- Показывает превью, фреймрейт, зоны триггера и отладочные окна.
- Используй для проверки корректности настроек.

## 10) Частые проблемы
- Failed to initialize screenshot source - источник захвата недоступен.
- Static dimension mismatch - размер входа не совпадает с моделью.
- Низкий FPS - уменьшить размер захвата, выключить дебаг, включить TRT.
