"""
Production settings - uses SQLite for simpler deployment
"""
from .settings import *

# Override database to use SQLite for production server
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Ensure DEBUG is False in production
DEBUG = False

# Add your server IP
ALLOWED_HOSTS = ['191.101.81.150', 'localhost', '127.0.0.1']

