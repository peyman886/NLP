from django.urls import path
from . import views
urlpatterns = [
    path('svm_home/', views.svmhome, name="grammarly-svmhome"),
    path('svm_home/get_subverb_match/', views.get_subverb_match, name="grammarly-subverbmatch"),
    path('syn_home/', views.synhome, name="grammarly-synhome"),
    path('syn_home/get_synonym/', views.get_synonym, name="grammarly-synonym"),
    path('grammar_home/', views.grammarhome, name="grammarly-grammarhome"),
    path('grammar_home/get_grammar_check/', views.get_grammar_check, name="grammarly-grammar"),
    path('predict_word/', views.get_next_word, name="grammarly-nextword"),
    path('', views.home, name="grammarly-home"),
]
