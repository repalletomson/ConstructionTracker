from django.urls import path
from . import views
from django.contrib import admin
from django.urls import path
from . import views  # Import your views here
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
  # Admin interface
    path('upload/', views.check_progress, name='upload_image'),  # Home page with upload form
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)  # Serve media files
# urlpatterns=[
#     path('',views.home,name='home'),
#     path('add/', views.show_addition_form, name='show_addition_form'),
#     path('check-progress/', views.check_progress, name='check_progress'),
# ] 


   
 