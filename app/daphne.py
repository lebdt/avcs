#!/Users/ebd/avcs/acvs_venv/bin/python

import os
import sys
import django
from subprocess import call

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'csv_processor.settings')

# Add the project directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Initialize Django
# django.setup()

# Now run Daphne
call(["daphne", "-b", "0.0.0.0", "-p", "8001", "csv_processor.asgi:application"])
