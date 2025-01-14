# NLP in Financial Market Analysis

.
├── client.py
├── Dockerfile
├── modules
│   ├── infer.py
│   └── train.py
├── poetry.lock
├── pyproject.toml
└── README.md

В рамках данного репозитория я сосредоточусь на Ops части своей дипломной работы - NLP in Financial Merket Analysis. Цель работы - с помощью NLP инструментов предсказывать конкретные финансовые метрики: долговую нагрузку, вероятность обвала индекса и т.д. Цель MLOps части - создать эффективный, шаблонный пайплайн обучения и выкатки моделей в prod.

### Данные

Задача достаточно специфичная, поэтому для нее не существует уже готового датасета, поэтому придется сделать его самостоятельно. В качестве основы возьмем финансовые новости, подойдет датасет от London Stock Exchange Group https://datasetsearch.research.google.com/search?src=2&query=Financial%20News%20Coverage&docid=L2cvMTF2amcwdnRrbA%3D%3D

Для предсказания конкретной финансовой метрики нам нужен датасет с ней, его можно получить используя API Yahoo Finance или из исторического датасета. В рамках этой работы я обращусь к данным NASDAQ-only торгуемых акций из https://datasetsearch.research.google.com/search?src=0&query=nasdaq%20stock%20performance&docid=L2cvMTFubnJkc2N0bQ%3D%3D

Имея датасеты новостей и предсказываемой метрики/флага/индекса с хронологическими метриками можно построить искомый датасет. Подводные камни: фильтрация только нужных новостей, поиск нужных рыночных срезов.

В качестве тестового датасета я взял уже проверенный моей предыдущей работой https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news/code. Он позволяет предсказывать настроение новости по ее тексту.

### Моделирование

Пока у меня нет финального решения для этой задачи, но в рамках этой работы в качестве бейзлайна я буду обучать/дообучать BERT подобные сети.

### Способ предсказания

По требованиям проекта был сделан клиент, поддерживающий опции обучения и инференса модели. Так же можно собрать docker образ с уже подготовленным в poetry окружением.
