import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin","django.contrib.auth","django.contrib.contenttypes",
    "django.contrib.sessions","django.contrib.messages","django.contrib.staticfiles",
    "rest_framework","chat","docs",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware","django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware","django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware","django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "chatserver.urls"
TEMPLATES = [{
    "BACKEND":"django.template.backends.django.DjangoTemplates","DIRS":[],
    "APP_DIRS":True,"OPTIONS":{"context_processors":[
        "django.template.context_processors.debug","django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth","django.contrib.messages.context_processors.messages",
    ]}}
]
WSGI_APPLICATION = "chatserver.wsgi.application"
if os.environ.get("DB_NAME"):
    DATABASES = {"default":{
        "ENGINE":"django.db.backends.postgresql",
        "NAME":os.environ["DB_NAME"],"USER":os.environ.get("DB_USER",""),
        "PASSWORD":os.environ.get("DB_PASSWORD",""),"HOST":os.environ.get("DB_HOST","localhost"),
        "PORT":os.environ.get("DB_PORT","5432"),"CONN_MAX_AGE":60}}
else:
    DATABASES = {"default":{"ENGINE":"django.db.backends.sqlite3","NAME": BASE_DIR / "db.sqlite3"}}
LANGUAGE_CODE = "en-us"; TIME_ZONE = "UTC"; USE_I18N = True; USE_TZ = True
STATIC_URL = "static/"; DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
