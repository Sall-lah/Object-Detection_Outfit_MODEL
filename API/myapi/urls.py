from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import get_users, create_user, user_details
from .scan_image import scan_image

urlpatterns = [
    # GET
    path("user/", get_users, name='get_users'),
    # POST
    path("user/create/", create_user, name='create_user'),
    # GET UPDATE DELETE
    path("user/<int:pk>/", user_details, name='user_details'),

    # SCAN POST
    path("scan", scan_image, name='scan_image')
] # + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)