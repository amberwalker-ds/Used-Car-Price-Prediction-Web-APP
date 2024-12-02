#txt and excel files from Riga tech Uni
data_2018_path = 'Data/1-s2.0-S2352340918316159-mmc2/2018-data.xlsx'
data_2019_path = 'Data/Dataset_Auto_2019.xlsx'
data_2020_path = 'Data/2020-data.xlsx'
file_path_txt_2021 = 'Data/Latvian-used_car_market_2021q1/*.txt'
file_path_txt_2022 = 'Data/Latvian Used-car Market Announcements Monitoring in 2022/*.txt'
file_path_txt_2023 = 'Data/Latvian Used-car Market Announcements Monitoring in 2023/*.txt'
cupra_path = 'Data/cupra_data.csv'

nfiles = [data_2018_path, data_2019_path, data_2020_path, file_path_txt_2021, 
         file_path_txt_2022, file_path_txt_2023]

#scripts/config.py
model_save_path = 'models/saved_xgb_model.json'
pipeline_save_path = 'models/saved_pipeline.pkl'

#test data
test_data = 'data/test_data.csv'

#target encoding smooting
smoothing = 0.2
#features to target encode
cat_features = ['make', 'model']
#features
features = ['make', 'model', 'engine', 'listing_year', 'car_age','fuel_bifuel', 
            'fuel_diesel','fuel_hybrid', 'fuel_petrol', 'is_luxury', 'mileage_bin_1',
            'mileage_bin_2', 'mileage_bin_3', 'mileage_bin_4', 'log_mileage', 
            'luxury_age_interaction', 'age_auto_interaction', 
            'age_engine_interaction','lux_auto_interaction', 'listing_month',
            'gearbox_type_automatic', 'is_suv_truck', 'is_reliable', 'popularity', 
            'country_czech republic', 'country_france', 'country_germany', 
            'country_italy', 'country_japan','country_romania', 'country_russia', 
            'country_south korea','country_spain', 'country_sweden', 'country_uk', 
            'country_ukraine','country_unknown', 'country_usa']
