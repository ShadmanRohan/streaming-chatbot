import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin","django.contrib.auth","django.contrib.contenttypes",
    "django.contrib.sessions","django.contrib.messages","django.contrib.staticfiles",
    "corsheaders",
    "rest_framework","chat",
]
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
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
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# OpenAI Configuration (API key from environment, never hardcoded)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Chat Configuration
CHAT_CONFIG = {
    "context_messages": int(os.environ.get("CHAT_CONTEXT_MESSAGES", "10")),
    "max_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "2000")),
    "temperature": float(os.environ.get("CHAT_TEMPERATURE", "0.7")),
}

# RAG Configuration
RAG_CONFIG = {
    "enabled": os.environ.get("RAG_ENABLED", "true").lower() == "true",
    "top_k": int(os.environ.get("RAG_TOP_K", "3")),
    "use_mmr": os.environ.get("RAG_USE_MMR", "true").lower() == "true",
}

# Memory Configuration (Phase 7)
MEMORY_CONFIG = {
    "max_tokens_context": int(os.environ.get("CHAT_MAX_TOKENS_CONTEXT", "3000")),
    "history_min_turns": int(os.environ.get("CHAT_HISTORY_MIN_TURNS", "6")),
    "summary_interval_turns": int(os.environ.get("SUMMARY_INTERVAL_TURNS", "5")),
}

# CORS Configuration (for browser demo)
CORS_ALLOW_ALL_ORIGINS = True  # For development only
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]
