# Introduction 
AVCS - Automatic Variable Creation System (Powered by IA)

# Getting Started
1.	Installation process
2.	Software dependencies

# Tips for getting it running

```sh
git clone https://github.com/lebdt/avcs.git
cd avcs
```
Setup a Python Virtual Environment: https://docs.python.org/3/library/venv.html

Example (Linux/Bash):

```sh
python -m venv avcs_venv
source avcs_venv/bin/activate
```

After activating the virtual environment, run:

```sh
pip install requirements.txt
```

# Run

To run the app after installing the required libraries run:

```sh
cd app
./daphne.py
```

or if it doesn't work, simply run:
```sh
cd app
daphne -b 0.0.0.0 -p 8001 csv_processor.asgi:application
```

# Collaborators

- Ednan Reidyner Garcia
- Eduardo Tadeu Domingues Rodrigues
- João Pedro da Silva Oliveira
- Lucas Duarte
