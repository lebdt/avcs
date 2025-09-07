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

# Set your OpenAI API Key

Create a file named `.env` under the `app` folder and then set your `OPENAI_API_KEY` as an environment variable for the project like so:

```sh
touch app/.env
echo OPENAI_API_KEY=xx-xxxx-xxxxxxxxxx > app/.env
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

# Demo

![Preview](./demo/avcs_demo.gif)

# Collaborators

- Ednan Reidyner Garcia
- Eduardo Tadeu Domingues Rodrigues
- João Pedro da Silva Oliveira
- Lucas Duarte
