# Liquidity Stress Index (LSI)

Система раннего предупреждения стресса ликвидности рублёвого денежного рынка на основе публичных данных ЦБ РФ, Минфина, Росказны и ФНС.

Проект автоматически загружает данные, рассчитывает сигналы по пяти модулям (M1–M5), агрегирует их в индекс **LSI (0–100)** и отображает результаты в Streamlit-дашборде с LLM-чатом.

## Шкала LSI

| LSI | Статус |
|---:|---|
| 0–39 | Норма |
| 40–69 | Напряжение |
| 70–100 | Стресс |

## Модули системы

### M1. Обязательные резервы (`src/modules/m1_reserves.py`)
- Данные: ЦБ РФ (резервы + RUONIA)
- Сигналы: спред фактических/обязательных резервов → MAD, RUONIA → MAD, флаг конца периода усреднения (6–10 число)

### M2. РЕПО ЦБ (`src/modules/m2_repo.py`)
- Данные: ЦБ РФ (требования по РЕПО, ключевая ставка)
- Сигналы: `cover_ratio_proxy` (текущий объём / 90-дневное среднее), `rate_spread_proxy` (7-дневная дельта / среднее), rolling MAD, флаг спроса

### M3. Размещение ОФЗ (`src/modules/m3_ofz.py`)
- Данные: Минфин (аукционы ОФЗ), fallback → ЦБ РФ, fallback → синтетический генератор
- Сигналы: cover ratio → MAD, yield spread → MAD, флаги недоспрос/переспрос

### M4. Налоговый календарь и сезонность (`src/modules/m4_tax.py`)
- Данные: ФНС (календарь), fallback → синтетический календарь (по правилам РФ)
- Сигналы: налоговая неделя, основной налоговый день, НДС, налог на прибыль, конец месяца/квартала, `seasonal_factor` (1.0–1.4, мультипликатор для LSI)

### M5. Казначейство и ликвидность (`src/modules/m5_treasury.py`)
- Данные: ЦБ РФ (дефицит/профицит ликвидности), Росказна (депозиты ЕКС)
- Сигналы: liquidity deficit → MAD, weekly/monthly delta → MAD, `num_placements` (число размещений депозитов Росказны) → MAD, флаг оттока

### Коррекция двойного счёта (`src/lsi.py:apply_double_count_correction`)
В налоговые периоды сигналы M1/M2/M5 могут дублироваться. Коррекция: вычисляется средняя премия модулей в налоговые недели/концы кварталов и вычитается из adjusted_score, чтобы отделить структурный стресс от календарного шума.

## ML-модель

`src/model.py` — RandomForestClassifier (class_weight='balanced', max_depth=5).

- Признаки: MAD-сигналы M1–M5, флаги, скользящие лаги
- Цель: `stress_target` (1 для калибровочных периодов: декабрь 2014, февраль–апрель 2022, август–сентябрь 2023)
- Результат: train ~96%, test ~89%
- Сохраняется: `models/lsi_model.pkl`, `data/final/lsi_ml.csv`, `data/final/feature_importance.csv`

## Синтетические данные (fallback)

При недоступности источников данные генерируются:

| Источник | Fallback |
|---|---|
| **Минфин (ОФЗ)** 503 | Генератор `generate_synthetic_ofz_auctions()` — 679 строк, среда каждой недели, 2014–2026 |
| **ФНС (календарь)** | Генератор `generate_synthetic_tax_calendar()` — 351 строка, правила РФ (НДС, прибыль, НДФЛ) |
| **Росказна (депозиты)** | Парсер русских дат + количество XML/docx как `num_placements` |

## LLM-чат

Дашборд содержит страницу «Аналитик (LLM-чат)» с интеграцией через **Ollama** (модель `qwen2.5-coder:7b`):
- Строит RAG-контекст из текущих данных (LSI, вклад модулей, флаги, тренды)
- Отвечает на естественно-языковые вопросы о ликвидности
- Fallback на keyword-ответы при недоступности Ollama

Требуется установленный и запущенный Ollama: `ollama pull qwen2.5-coder:7b`

## Структура проекта

```text
project/
├── data/
│   ├── raw/                  # исходные загруженные данные (CSV)
│   └── final/                # финальные таблицы LSI, ML, отчёты
├── models/
│   └── lsi_model.pkl         # обученная ML-модель
├── src/
│   ├── loaders/              # загрузчики данных
│   │   ├── base.py           # BaseLoader (HTTP, Excel, HTML)
│   │   ├── cbr_loader.py     # ЦБ РФ
│   │   ├── fns_loader.py     # ФНС + синтетический fallback
│   │   ├── minfin_loader.py  # Минфин + синтетический fallback
│   │   └── treasury_loader.py# Росказна + парсер
│   ├── modules/              # расчёт M1–M5
│   │   ├── common.py
│   │   ├── m1_reserves.py
│   │   ├── m2_repo.py
│   │   ├── m3_ofz.py
│   │   ├── m4_tax.py
│   │   └── m5_treasury.py
│   ├── normalization.py      # скользящее MAD (3 года)
│   ├── lsi.py                # формульный LSI + double-count correction
│   ├── model.py              # RandomForestClassifier
│   ├── backtest.py           # проверка на стресс-периодах + holdout
│   ├── sensitivity.py        # анализ чувствительности ±20%
│   └── dashboard.py          # Streamlit + LLM-чат (Ollama)
├── download_data.py           # загрузка данных
├── load_data.py               # расчёт M1–M5 и LSI
├── run_all.py                 # полный пайплайн
├── requirements.txt
└── README.md
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для LLM-чата (опционально):
```bash
# Установите Ollama: https://ollama.ai
ollama pull qwen2.5-coder:7b
# Убедитесь, что Ollama запущен
ollama serve
```

## Запуск

Полный пайплайн одной командой:
```bash
python3 run_all.py
```

Затем дашборд:
```bash
streamlit run src/dashboard.py
```

Поэтапно:
```bash
python3 download_data.py          # → data/raw/
python3 load_data.py              # → data/final/lsi.csv
python3 src/model.py              # → models/lsi_model.pkl, lsi_ml.csv
python3 src/backtest.py           # → data/final/backtest_report.csv
python3 src/sensitivity.py        # → data/final/sensitivity_report.csv
streamlit run src/dashboard.py    # → http://localhost:8501
```

## Дашборд

Страницы:
1. **Дашборд** — LSI, алерты, флаги, вклад модулей, графики M1–M5, feature importance, backtest, sensitivity, автокомментарий (через LLM)
2. **Аналитик (LLM-чат)** — RAG-чат по данным системы на русском языке

## Выходные файлы

| Файл | Описание |
|---|---|
| `data/final/lsi.csv` | Формульный LSI + все сигналы модулей |
| `data/final/lsi_ml.csv` | ML LSI + stress_probability |
| `data/final/feature_importance.csv` | Важность признаков |
| `data/final/backtest_report.csv` | Backtest калибровка + holdout |
| `data/final/sensitivity_report.csv` | Чувствительность ±20% весов |
| `models/lsi_model.pkl` | Обученная RandomForest |

## Ограничения

- Минфин (minfin.gov.ru) возвращает 503 — ОФЗ работают на синтетических данных
- ФНС (nalog.gov.ru) не отдаёт таблицу — календарь синтетический
- Росказна (roskazna.gov.ru) отдаёт только имена файлов, не суммы депозитов — proxy через `num_placements`
- Качество ML LSI зависит от разметки стрессовых периодов
- Структура HTML источников может меняться
