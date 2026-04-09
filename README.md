# ai-fashion-trends

Проект для анализа модных трендов из текста и изображений.

Референсы:
- https://www.wgsn.com/en
- https://heuritech.com/company-about-us/

## Реализация

- Обработка текстовых данных: сбор, извлечение трендов через LLM и формирование словаря тегов.
- Анализ изображений: классификация макияжа (smoky eyes / blue eyeshadow / other) с демо-страницей.
- Прогноз временных рядов: недельные ряды трендов и прогноз (ETS/Holt-Winters с fallback на Theta и drift).
- Демо в Streamlit: визуализация метрик и графиков по текстовым трендам + отдельная страница классификатора изображений.

## Запуск

Требуется Python 3.10+.

```bash
cd /Users/vorudnikova/Desktop/fashion/ai-fashion-trends
uv sync
```

### 1) Сгенерировать артефакты прогноза для демо

```bash
uv run python -m ai_fashion_trends synthetic-forecast
```

### 2) Обучить классификатор макияжа (если есть папки с изображениями)

Ожидаемые папки:
- `data/makeup_dataset/smoky_eyes/`
- `data/makeup_dataset/blue-eyeshadow/` (или `blue_eyeshadow`)
- `data/makeup_dataset/other/`

Быстрый вариант:

```bash
uv run python -m ai_fashion_trends train-makeup-classifier --no-mediapipe
```

Вариант с дообучением CNN (ResNet18):

```bash
uv sync --extra torch
uv run python -m ai_fashion_trends train-makeup-cnn --no-mediapipe
```

### 3) Запустить демо

```bash
uv run streamlit run streamlit_app.py
```

## Полезные команды

```bash
uv run python -m ai_fashion_trends --help
```

Собрать словарь тегов из извлеченных трендов:

```bash
uv run python -m ai_fashion_trends build-tag-dictionary
```
