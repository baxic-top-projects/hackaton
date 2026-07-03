# Фабрика гипотез

Streamlit-прототип для задачи Норникель AI Hackathon: гибридная система на стыке GraphRAG, Long-Context LLM и внешних символьных/физических калькуляторов.

## Что делает приложение

- Принимает цель/KPI, ограничения, доступное сырье, оборудование и веса критериев ранжирования.
- Загружает базу знаний из `txt`, `md`, `pdf`, `docx`, `csv`, `xlsx`, `png`, `jpg`, `jpeg`, `webp`.
- Разбивает документы на фрагменты, строит устойчивый hashing retrieval индекс и GraphRAG-граф материалов, процессов, свойств, параметров и источников.
- Запускает гибридный пайплайн: ingestion, GraphRAG retrieval, генерация гипотез, расчетная валидация, подготовка long-context для LLM.
- Генерирует проверяемые гипотезы по шаблонам материаловедения и металлургических процессов.
- Проверяет гипотезы внешними калькуляторами: стоимость легирования, наличие оборудования, прогноз KPI-эффекта, термоокно, баланс процесса.
- Выполняет контрфактуальный анализ: baseline, сценарий без ключевого фактора и замена процесса ближайшей альтернативой.
- Запускает предсказательную surrogate-модель `RandomForestRegressor` для оценки ожидаемого KPI uplift.
- Извлекает OCR-текст из изображений при наличии Tesseract (`rus`, `eng`, `chi_sim`); без OCR учитывает изображения как визуальные источники с метаданными.
- Нормализует английские и китайские доменные термины в русскую онтологию (`tailings`, `flotation`, `尾矿`, `浮选` и др.).
- Извлекает метаданные из источников: даты, авторов, условия экспериментов (`pH`, температура, проценты, крупность).
- Оценивает гипотезы по новизне, реализуемости, ожидаемой ценности, риску и уверенности источников.
- Выполняет внешнюю проверку новизны по Crossref, Semantic Scholar и PatentsView с кэшированием результата.
- Обучает легкую ML-модель на экспертном feedback и использует ее при следующих ранжированиях.
- Показывает граф связей "гипотеза - фактор - источник".
- Сохраняет GraphRAG-граф в `JSON`, `GraphML`, `RDF/Turtle` и опционально синхронизирует его в Neo4j.
- Экспортирует результат в CSV, JSON, Markdown, DOCX и PDF-отчет.
- Сохраняет экспертную обратную связь по гипотезам в `data/feedback.json`.
- Предоставляет HTTP API `/api/generate`, role-based token auth и отправку гипотез в Jira/YouTrack.

## Быстрый запуск

## Публичное демо

- UI: `https://hackaton.baxic.ru`
- API: `https://hypothesisapi.baxic.ru`
- API Docs: `https://hypothesisapi.baxic.ru/docs`
- Healthcheck: `https://hypothesisapi.baxic.ru/health`

Если задан `API_AUTH_TOKEN`, API-запросы должны передавать заголовок `X-API-Key`.

Реальные токены не хранятся в README и задаются в `.env` на сервере:

```env
APP_AUTH_TOKEN=<UI access token>
API_AUTH_TOKEN=<API access token>
# опционально несколько ролей:
APP_TOKENS=viewer-token:viewer,expert-token:expert,admin-token:admin
API_TOKENS=reader-token:viewer,research-token:researcher,admin-token:admin
```

Пример:

```bash
curl -X POST https://hypothesisapi.baxic.ru/api/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_AUTH_TOKEN>" \
  -d '{"target":"снизить потери Ni в хвостах","limit":3}'
```

## Локальный запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Если пользователь не загрузит файлы, приложение использует встроенную демо-базу `data/sample_knowledge`.

## Запуск в Docker

Создайте `.env` на основе `.env.example`, затем запустите:

```powershell
docker compose up --build
```

Приложение будет доступно на `http://localhost:8502`. Compose пробрасывает `.env` внутрь контейнера, монтирует `./data` в `/app/data` и поднимает Neo4j для синхронизации графа.

HTTP API поднимается вторым сервисом на `http://localhost:8503`:

```powershell
curl http://localhost:8503/health
```

Neo4j Browser доступен на `http://localhost:7474`, Bolt endpoint — `bolt://localhost:7687`.

Пример генерации через API:

```powershell
curl -X POST http://localhost:8503/api/generate `
  -H "Content-Type: application/json" `
  -d "{\"target\":\"снизить потери Ni в хвостах\",\"constraints\":\"лабораторная проверка 2 недели\",\"documents\":[{\"source\":\"case.txt\",\"text\":\"tailings flotation pH recovery losses\"}]}"
```

## Опционально: Yandex AI Studio

Не храните API-ключ в репозитории. Перед запуском задайте переменные окружения:

```powershell
$env:YANDEX_API_KEY="ваш_api_ключ"
$env:YANDEX_FOLDER_ID="ваш_folder_id"
# Необязательно: $env:YANDEX_MODEL="yandexgpt/latest"
python app.py
```

Если переменные заданы, приложение автоматически добавляет long-context сводку YandexGPT поверх GraphRAG-контекста, гипотез и расчетных проверок. Без этих переменных приложение работает в офлайн-режиме.

## Интеграции

Для прямого создания задач задайте переменные в `.env`.

Jira Cloud:

```powershell
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=user@example.com
JIRA_API_TOKEN=your_token
JIRA_PROJECT_KEY=LAB
```

YouTrack:

```powershell
YOUTRACK_BASE_URL=https://youtrack.example.com
YOUTRACK_TOKEN=perm:...
YOUTRACK_PROJECT_ID=0-0
```

Внешнюю проверку новизны можно отключить для полностью офлайн-демо:

```powershell
NOVELTY_CHECK_ENABLED=0
```

Опциональная авторизация:

```powershell
APP_AUTH_TOKEN=local-ui-token
API_AUTH_TOKEN=external-api-token
# или несколько токенов с ролями
APP_TOKENS=viewer-token:viewer,expert-token:expert,admin-token:admin
API_TOKENS=reader-token:viewer,research-token:researcher,admin-token:admin
```

При заданном `API_AUTH_TOKEN` внешние клиенты должны передавать заголовок `X-API-Key`.

Опциональная синхронизация графа в Neo4j:

```powershell
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

## Демо-сценарий

1. Запустите `python app.py`.
2. При необходимости загрузите свои документы, таблицы или изображения через боковую панель.
3. Нажмите "Сгенерировать гипотезы".
4. Покажите жюри GraphRAG-метрики, таблицу ранжирования, карточки гипотез, цитаты источников, калькуляторы и дорожную карту проверки.
5. Скачайте PDF/DOCX/Markdown-отчет или JSON как пример интеграции с внешними системами.

## Архитектура

```mermaid
flowchart LR
    user["Исследователь"] --> ui["Streamlit UI"]
    ui --> ingestion["Парсинг файлов"]
    ingestion --> textIndex["Hashing retrieval индекс"]
    ingestion --> graphIndex["GraphRAG граф сущностей и параметров"]
    textIndex --> graphRetrieval["GraphRAG retrieval"]
    graphIndex --> graphStore["JSON/GraphML/RDF + Neo4j query"]
    graphIndex --> graphRetrieval
    graphRetrieval --> generator["Генерация гипотез"]
    generator --> counterfactual["Counterfactual analysis"]
    counterfactual --> calculators["Символьные и физические калькуляторы"]
    calculators --> novelty["Crossref/S2/Patents novelty"]
    calculators --> predictive["Predictive KPI model"]
    predictive --> feedback["ML feedback reranking"]
    feedback --> llmContext["Long-context пакет"]
    llmContext --> yandex["YandexGPT"]
    feedback --> cards["Карточки гипотез"]
    cards --> exports["CSV/JSON/Markdown/DOCX/PDF"]
    cards --> trackers["Jira/YouTrack/API"]
```

## Почему это подходит под кейс

Решение не просто генерирует текст, а сохраняет трассировку к источникам и промежуточным расчетам: каждая гипотеза содержит цитаты, GraphRAG-обоснование, механизм влияния, расчетные проверки, прогнозный диапазон KPI-эффекта, риски, ресурсы и план проверки. Веса критериев и экспертная обратная связь помогают менять стратегию ранжирования под конкретную лабораторию, бюджет или горизонт проверки.

Базовый прототип работает локально и не требует внешних API. Это важно для конфиденциальных промышленных данных и для стабильного демо. Опциональные внешние слои Yandex AI Studio, Crossref/Semantic Scholar/PatentsView, Neo4j и Jira/YouTrack подключаются через `.env`, но не заменяют GraphRAG/scoring/calculators и не ломают офлайн-режим.

## Соответствие требованиям кейса

Проект закрывает требования кейса на уровне хакатонного MVP: это комплексная "Фабрика гипотез", а не только RAG-чат.

- Полезность для исследователей: гипотезы конкретные, проверяемые, с механизмом влияния, ресурсами, рисками и дорожной картой проверки.
- Прозрачность и обоснованность: есть цитаты источников, GraphRAG-связи, объяснимые пути в графе, расчетные проверки, novelty-check и counterfactual explanation.
- Гибкость входных данных: поддержаны `txt/md/pdf/docx/csv/xlsx/png/jpg/jpeg/webp`, OCR, fallback для шумных данных, метаданные и мультиязычная нормализация.
- Масштабируемость: архитектура модульная — ingestion, retrieval, GraphRAG, calculators, counterfactual, predictive KPI, novelty, feedback, exporters, API. Граф хранится в `JSON/GraphML/RDF` и Neo4j.
- Интеграция: есть публичный API, Swagger, CSV/JSON, PDF/DOCX, Jira/YouTrack, Docker Compose и Neo4j.

Функциональные требования также закрыты:

- Прием и предобработка данных: парсинг документов, таблиц, изображений, OCR, метаданные — даты, авторы, условия экспериментов, `pH`, температура, проценты, крупность.
- Генерация гипотез: извлечение сущностей и связей, паттерны, контрфактуальный анализ, predictive KPI model и ranking.
- Обоснование и визуализация: карточки гипотез, граф связей, GraphRAG-контекст, источники, цитаты, риски, uncertainty через confidence/risk/novelty.
- Экспорт и интеграция: PDF/DOCX, CSV/JSON/Markdown, API, Jira/YouTrack, feedback learning.

Нефункциональные требования:

- Интерпретируемость: каждый шаг виден через agent trace, источники, граф, калькуляторы, novelty и counterfactual.
- Мультиязычность: OCR `rus+eng+chi_sim`, нормализация английских и китайских терминов в русскую онтологию.
- Надежность: fallback для пустых и шумных данных, `HashingVectorizer`, офлайн-режим без внешних API.
- Производительность: генерация рассчитана на интерактивный режим, без тяжелых моделей в основном цикле.
- Безопасность: локальный Docker-контур, UI token, API token, role-based access, Neo4j во внутренней сети.

Ограничения MVP честно обозначены: novelty screening не заменяет промышленную патентную экспертизу, predictive KPI surrogate не является полноценной физико-химической симуляцией, token-based RBAC не заменяет корпоративный SSO. Для хакатонного решения эти ограничения компенсируются интерпретируемостью, модульностью и готовностью к интеграции.

Короткая формулировка для защиты:

> Мы реализовали гибридную агентную фабрику гипотез: GraphRAG + Long-Context LLM + counterfactual analysis + predictive KPI model + novelty screening + expert feedback learning, с экспортом в отчеты, API, таск-трекеры и Neo4j-backed graph storage.
