from django.urls import path
from . import views

app_name = "dunnhumby_site"

urlpatterns = [
    path("", views.site_index, name="index"),
    path("basket-analysis/", views.basket_analysis, name="basket_analysis"),
    path("association-rules/", views.association_rules, name="association_rules"),
    path("customer-segments/", views.customer_segments, name="customer_segments"),
    path("data-management/", views.data_management, name="data_management"),
    # JSON/API endpoints used by front-end JS
    path("api/table/", views.api_get_table_data, name="api_get_table_data"),
    path("api/update/", views.api_update_record, name="api_update_record"),
    path("api/export/", views.api_export_data, name="api_export_data"),
    path("api/schema/", views.api_table_schema, name="api_table_schema"),
    path("api/basket/", views.api_basket_details, name="api_basket_details"),
    path("api/product/", views.api_product_details, name="api_product_details"),
    path("api/household/", views.api_household_details, name="api_household_details"),
    path("api/segment/", views.api_segment_details, name="api_segment_details"),
    # ML API endpoints
    path("api/ml/predictive/", views.predictive_analysis_api, name="predictive_analysis_api"),
    path("api/ml/train/", views.train_ml_models, name="train_ml_models"),
    path("api/ml/predictions/", views.get_predictions, name="get_predictions"),
    path("api/ml/recommendations/", views.get_recommendations, name="get_recommendations"),
    path("api/ml/performance/", views.get_model_performance, name="get_model_performance"),
    path("api/ml/training-status/", views.training_status_api, name="training_status_api"),
]
