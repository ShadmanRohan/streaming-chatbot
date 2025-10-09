from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.conf.urls.static import static
import os

def health(_): return JsonResponse({"status": "ok"})

def sse_demo(request):
    """Serve the SSE demo HTML file"""
    demo_path = os.path.join(settings.BASE_DIR, 'static', 'sse-demo.html')
    with open(demo_path, 'r') as f:
        content = f.read()
    return HttpResponse(content, content_type='text/html')

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health/", health),
    path("api/", include("chat.urls")),
    path("demo/", sse_demo, name="sse-demo"),
]

# Serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")
