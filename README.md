# Классификация и преобразование стиля русскоязычных текстов

Код к магистерской диссертации «Автоматическая классификация научного и научно-популярного стилей русскоязычных текстов по компьютерным наукам с применением в задаче преобразования стиля».

## Структура

```
├── data/               # data500.csv (не включён)
├── src/                # модули
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── metrics.py
│   └── models/
│       ├── baseline.py
│       ├── svm.py
│       ├── bilstm.py
│       ├── rubert.py
│       └── generator.py
└── scripts/
    ├── validate_dataset.py
    ├── train_classifiers.py
    └── train_generator.py
```

## Запуск

```bash
pip install -r requirements.txt
python scripts/validate_dataset.py
python scripts/train_classifiers.py
python scripts/train_generator.py
```

## Данные

Параллельный корпус (500 пар): `data/data500.csv` — CSV без заголовка, три столбца: `text`, `label` (0 — науч.-поп., 1 — научный), `pair_id`.

## Модели

| Модель           | Тип                |
|------------------|--------------------|
| Baseline         | LogReg + признаки  |
| SVM              | TF-IDF + LinearSVC |
| Char-CNN-BiLSTM  | Посимвольная сеть  |
| RuBERT           | Трансформер        |
| ruT5             | Генератор стиля    |
