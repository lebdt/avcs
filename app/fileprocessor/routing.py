from django.urls import re_path
from . import receiver

websocket_urlpatterns = [
    re_path(r'ws/csv/$', receiver.CSVConsumer.as_asgi()),
]
