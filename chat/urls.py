from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ChatSessionViewSet, retrieve_chunks

router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='document')
router.register(r'sessions', ChatSessionViewSet, basename='session')

urlpatterns = [
    path('', include(router.urls)),
    path('retrieve/', retrieve_chunks, name='retrieve'),
]
