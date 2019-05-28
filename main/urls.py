from django.urls import path

from main.views import ImageView

urlpatterns = [
    path('', ImageView.as_view())
]
