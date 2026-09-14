# OCR PDF скрининга

Streamlit-приложение для загрузки PDF скрининга, OCR-распознавания, извлечения ключевых признаков и предсказания осложнений по извлечённым данным. Результаты можно экспортировать в CSV/JSON. Точка входа — `main.py`; основные модули находятся в `app/`: конфигурация, OCR, parser, predictor и UI. Зависимости описаны в `pyproject.toml`. fileciteturn2file0L2-L2

## Запуск приложения

```bash
uv run streamlit run main.py
```

## Практическая работа 1: Linux и Bash

Linux-часть проекта добавляет автоматизацию через `scripts/project.sh` и `Makefile`.

### Структура

```text
preg-comp/
├── app/                 # OCR, parser, predictor и UI
├── data/                # локальные данные, не хранятся в Git
├── artifacts/           # локальные ML-артефакты
├── logs/                # журналы запуска
├── results/             # результаты работы
├── tmp/                 # временные файлы
├── scripts/
│   └── project.sh       # Bash-автоматизация
├── main.py
├── pyproject.toml
├── uv.lock
├── Makefile
└── .dockerignore
```

`data/` и `artifacts/` уже исключены из Git, поэтому крупные локальные данные и ML-артефакты не попадают в репозиторий. fileciteturn4file0L2-L2

### Bash-скрипт

Сделать скрипт исполняемым:

```bash
chmod +x scripts/project.sh
```

Создать рабочие каталоги:

```bash
bash scripts/project.sh init
```

Запустить приложение в foreground:

```bash
PORT=8501 bash scripts/project.sh run
```

Запустить приложение в фоне:

```bash
PORT=8501 bash scripts/project.sh start
```

Проверить процесс и PID:

```bash
bash scripts/project.sh status
ps aux | grep streamlit
pgrep -af 'streamlit run main.py'
```

Остановить фоновый процесс:

```bash
bash scripts/project.sh stop
```

Найти Python-файлы:

```bash
bash scripts/project.sh find '*.py'
```

Найти текст в исходниках:

```bash
bash scripts/project.sh grep 'predict'
```

Проанализировать исходники и логи:

```bash
bash scripts/project.sh analyze
```

Проверить Linux и дисковое пространство:

```bash
bash scripts/project.sh system
```

Очистить генерируемые данные:

```bash
bash scripts/project.sh clean
```

### Полезные команды Linux

```bash
pwd
ls -lah
cd app
cd ..
mkdir -p results/archive
touch results/example.txt
cp results/example.txt results/archive/
mv results/archive/example.txt results/example-copy.txt
rm results/example-copy.txt
cat README.md
head -n 20 README.md
tail -n 20 logs/streamlit.log
wc -l app/*.py
sort logs/streamlit.log
find . -type f -name '*.py'
grep -RIn 'predict' app

du -sh .
du -sh app data artifacts
df -h .
uname -a
whoami
command -v python
command -v uv
command -v tmux
env | sort | head
```

`du` показывает занятое файлами место конкретного каталога, а `df` — использование и свободное место файловой системы, содержащей каталог.

### Работа с процессами

Для демонстрации длительного процесса:

```bash
make start
make status
jobs -l
ps -ef | grep streamlit
pgrep -af streamlit
kill <PID>
```

`&` запускает команду в фоне текущей shell-сессии. `jobs` показывает фоновые jobs этой shell, `ps` — процессы, `pgrep` ищет процессы по имени/командной строке, а `kill` отправляет процессу сигнал.

### tmux

Создать сессию с приложением:

```bash
make tmux
```

Посмотреть сессии:

```bash
tmux ls
```

Подключиться:

```bash
tmux attach -t preg-comp
```

Отсоединиться без остановки процесса: `Ctrl+B`, затем `D`.

Завершить сессию:

```bash
tmux kill-session -t preg-comp
```

### Makefile

```bash
make help
make init
make start
make status
make stop
make run
make find PATTERN='*.py'
make grep PATTERN='predict'
make analyze
make system
make tmux
make clean
```

Переменная `PORT` позволяет изменить порт без изменения исходников:

```bash
make start PORT=8502
```

## Минимальный сценарий демонстрации

```bash
make init
make start
make status
make find PATTERN='*.py'
make grep PATTERN='streamlit'
make analyze
make system
make stop
```

Затем отдельно показать `tmux ls`, `tmux attach -t preg-comp`, `du -sh .`, `df -h .`, `find`, `grep`, `ps`, `pgrep` и `kill`.

## Теоретический минимум

1. Абсолютный путь начинается от `/`, относительный — считается от текущего каталога. `.` обозначает текущий каталог, `..` — родительский.
2. `pwd` показывает текущий каталог; `cd` меняет каталог; `ls` показывает содержимое; `mkdir` создаёт каталоги; `cp` копирует; `mv` перемещает или переименовывает; `rm` удаляет; `touch` создаёт файл или обновляет его время.
3. `find` ищет файлы и каталоги по атрибутам файловой системы, а `grep` ищет текст внутри файлов.
4. `head` и `tail` показывают начало и конец файла; `wc` считает строки/слова/байты; `sort` сортирует строки. Они удобны для быстрой работы с логами и результатами.
5. `du` измеряет использование места каталогами/файлами, `df` показывает свободное и занятое место файловой системы.
6. Переменная окружения — значение, доступное процессу через окружение. `export` передаёт shell-переменную дочерним процессам. `PATH` содержит каталоги, где shell ищет исполняемые программы; `command -v` показывает найденную команду.
7. Процесс — выполняющаяся программа. PID — уникальный идентификатор процесса. Его можно найти через `ps` или `pgrep`.
8. `&` запускает команду в фоне. `jobs` работает с jobs текущей shell, `ps` показывает процессы, `pgrep` ищет процессы, `top/htop` позволяют наблюдать нагрузку, `kill` отправляет сигнал процессу.
9. `tmux` сохраняет терминальную сессию после отключения SSH/терминала, поэтому длительный запуск приложения не зависит от открытого окна терминала.
10. Bash-скрипт — текстовый файл с командами shell. Shebang `#!/usr/bin/env bash` указывает интерпретатор. Параметры доступны через `$1`, `$2` и т.д.; `$@` содержит все параметры.
11. Переменные хранят значения, `if` выполняет команды по условию, циклы `for/while` позволяют повторять операции. В `project.sh` это используется для проверки PID и поиска/обработки файлов.
12. Makefile содержит targets — именованные сценарии. `make start`, например, заменяет ручной вызов Bash-скрипта с нужными параметрами.
13. В проекте автоматизированы подготовка каталогов, запуск Streamlit, работа с PID, поиск файлов и текста, анализ логов, проверка ресурсов Linux и очистка временных результатов. Это выбрано потому, что именно эти операции регулярно нужны при запуске и демонстрации ML-приложения.
