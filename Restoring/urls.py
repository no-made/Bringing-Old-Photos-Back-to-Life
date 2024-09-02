
from django.urls import include, path
from . import views
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static
router = routers.DefaultRouter()
urlpatterns = [
    path('', views.index, name='index'),
    path('api/', include(router.urls)),
    path('landing', views.landing, name='landing'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + \
              static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT[0])
