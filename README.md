<div align="center">

# ZTXAI
[![Python](https://img.shields.io/badge/Python-3.10%2B-FFD43B?logo=python)](#)
[![Windows](https://img.shields.io/badge/Windows-10%2F11-0078D6?logo=windows)](#)
[![GUI](https://img.shields.io/badge/GUI-DearPyGui-2ea44f)](#)
</div>

## Overview ✨
ZTXAI - AI-ассистент на базе YOLO с GUI на DearPyGui. Основной вход `main.py`, логика разделена на `src/` и `core/`.

> [!WARNING]
> Используйте на свой риск.

> [!NOTE]
> Рекомендуется видеокарта уровня RTX 20xx и выше.

## Quick Start 🚀
1) Установка зависимостей:
```
install.bat
```
2) Запуск:
```
run.bat
```

## Project Layout 📁
- `main.py` - точка входа.
- `src/` - рабочие модули (захват, инференс, PID, профайлер).
- `core/` - GUI, конфиг, модель, логика наведения.
- `makcu/` - драйвер/интерфейс устройства ввода.
- `cfg.json` - пользовательский конфиг.
- `requirements.txt` - зависимости Python.
- `run.bat` / `install.bat` / `build.bat` / `profile.bat` - утилиты запуска.

## Build 🧱
```
build.bat
```
Результат: `output/main/main.exe`

## Profiling 🔎
```
profile.bat
```
Скрипт спросит, ставить ли `py-spy`, и запустит профайлер. Профайл нужен, чтобы увидеть, где тратится CPU и что тормозит в Python-коде.

## Models 🤖
- Поддержка ONNX и TensorRT (`.engine`).
- Для TRT `.engine` создается при необходимости.
- Размер входа берется из модели/engine.

## Capture 🎥
- Источники: Standard (экран), OBS, Capture Card.
- Выбор источника задается в GUI.
- При ошибках захвата проверяй настройки и доступность источника.

## Config & Classes 🧩
- Основной конфиг: `cfg.json`.
- Имена классов задаются файлом или вручную (GUI).
- Количество классов определяется моделью и/или списком имен.

## Documentation ??
- ?????? ???????? ???? ? ??????????: `docs/HELP.md`

## Troubleshooting 🧯
- `Failed to initialize screenshot source` - проверь выбранный источник захвата.
- `Static dimension mismatch` - размер входа модели не совпадает с ожиданиями.
- При падениях смотри `error_log.txt`.
