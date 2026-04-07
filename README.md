# ai-fashion-trends

Прогнозирование модных трендов с помощью ИИ-агентов.

Референсы:
https://www.wgsn.com/en
https://heuritech.com/company-about-us/

## Интерфейс демо

**Демонстрационный UI делаем на [Streamlit](https://streamlit.io/)**. 

- Точка входа демо: файл **`streamlit_app.py`** в корне репозитория.
- Логику загрузки данных и расчёты по возможности выносим в пакет **`ai_fashion_trends`** (`src/…`), а Streamlit только вызывает эти функции и рисует виджеты.

## Структура репозитория

```
ai-fashion-trends/
├── README.md
├── pyproject.toml          # метаданные пакета, зависимости (в т.ч. streamlit)
├── streamlit_app.py        # демо UI (Streamlit)
├── data/                   # сырые и подготовленные данные (пока пусто)
└── src/
    └── ai_fashion_trends/  # код приложения (данные, модели — позже)
        ├── __init__.py
        └── __main__.py     # точка входа CLI
```

Пайплайны обработки данных в пакете пока не реализованы.

## Установка

Нужен **Python 3.10+**.

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

После установки доступны зависимости проекта, включая **Streamlit**.

## Запуск

### Демо в браузере (Streamlit)

Из корня репозитория, с активированным venv:

```bash
streamlit run streamlit_app.py
```

**Важно:** сначала `pip install -e .`, чтобы импорт `ai_fashion_trends` из `streamlit_app.py` работал.

### CLI (каркас)

```bash
python -m ai_fashion_trends
```

## To-do

### Данные

TBD

### Демо (Streamlit)

TBD
