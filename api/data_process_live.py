def create_is_luxury_column(df):
    """
    Creates an 'is_luxury' indicator column based on brand and model.

    Args:
        df: The DataFrame containing car data.
        luxury_brands: A list of luxury car brands.
        luxury_models: A list of luxury car models (from non-luxury brands).

    Returns:
        The DataFrame with the 'is_luxury' column added.
    """
    #luxury column
    luxury_brands = [
        'astonmartin', 'audi', 'bentley', 'bmw', 'bugatti', 'cadillac', 'ferrari', 'jaguar',
        'lamborghini', 'landrover','rover', 'lexus', 'maserati', 'maybach', 'mclaren', 'mercedes',
        'porsche', 'rollsroyce', 'tesla', 'genesis', 'alfaromeo', 'infiniti', 'acura', 'lincoln',
        'volvo', 'polestar', 'gaz', 'aurus', 'ziL', 'volga', 'rangerover', 'koenigsegg', 'pagani',
        'rimac']

    luxury_models = [
        'phaeton', 'touareg', 'arteon', 'vignale', 'mazda6 signature', 'signature',
        'avalon', 'tlx', 'mdx', 'q50', 'qx50', 'g70', 'g80', 'g90', 'k900', 'stinger',
        'volvo s90', 'volvo xc90', 'maxima', 'platinum', 'xle', 'limited',
        'touring', 'calligraphy', 'highlanderplatinum'
    ]

    df['is_luxury'] = 0  # Initialize to 0

    # Mark luxury brands as luxury
    df.loc[df['make'].isin(luxury_brands), 'is_luxury'] = 1

    # Mark luxury models from non-luxury brands as luxury
    df.loc[(df['model'].isin(luxury_models)) & (df['is_luxury'] == 0), 'is_luxury'] = 1
    #This was edited to ensure that if the brand was luxury and the model is not in the luxury model list, it is not overwritten to 0

    return df

#function to add feaetures
def add_luxury_and_popularity_features(df):  
    # List of makes and models that are commonly SUVs or trucks
    suv_truck_models = {
        'toyota': ['land cruiser', '4runner', 'highlander', 'tacoma', 'tundra'],
        'ford': ['explorer', 'expedition', 'f-150', 'bronco'],
        'chevrolet': ['tahoe', 'suburban', 'silverado'],
        'nissan': ['patrol', 'xterra', 'armada', 'titan'],
        'jeep': ['wrangler', 'cherokee', 'grand cherokee'],
        # Add more make and model combinations as needed
    }

    # List of reliable makes or specific models
    reliable_models = {
        'toyota': ['corolla', 'camry', 'land cruiser'],
        'honda': ['civic', 'accord', 'cr-v'],
        'subaru': ['outback', 'forester'],
        # Add more reliable makes and models as needed
    }

        # Define country of origin for each make
    country_of_origin = {
        'audi': 'germany', 
        'bmw': 'germany', 
        'mercedes': 'germany', 
        'volkswagen': 'germany', 
        'porsche': 'germany', 
        'skoda': 'czech republic', 
        'vaz': 'russia', 
        'volvo': 'sweden', 
        'chevrolet': 'usa', 
        'cadillac': 'usa', 
        'chrysler': 'usa', 
        'dodge': 'usa', 
        'ford': 'usa', 
        'jeep': 'usa', 
        'lincoln': 'usa', 
        'pontiac': 'usa', 
        'toyota': 'japan', 
        'lexus': 'japan', 
        'honda': 'japan', 
        'infiniti': 'japan', 
        'mazda': 'japan', 
        'mitsubishi': 'japan', 
        'nissan': 'japan', 
        'subaru': 'japan', 
        'suzuki': 'japan', 
        'daihatsu': 'japan', 
        'hyundai': 'south korea', 
        'kia': 'south korea', 
        'ssangyong': 'south korea', 
        'citroen': 'france', 
        'peugeot': 'france', 
        'renault': 'france', 
        'fiat': 'italy', 
        'lancia': 'italy', 
        'alfaromeo': 'italy', 
        'jaguar': 'uk', 
        'landrover': 'uk', 
        'mini': 'uk', 
        'seat': 'spain', 
        'gaz': 'russia', 
        'moskvich': 'russia', 
        'uaz': 'russia', 
        'zaz': 'ukraine', 
        'smart': 'germany', 
        'dacia': 'romania', 
        'hummer': 'usa', 
        'izh': 'russia', 
        'cita': 'unknown',  # Unknown; adjust if more information is available
        'cupra': 'spain', 
        'opel': 'germany', 
        'saab': 'sweden',
        'daewoo': 'south korea', 
        'isuzu': 'japan', 
        'other': 'unknown'  # Adjust if you can assign specific countries later
    }
    
    # SUV or truck indicator
    def is_suv_or_truck(row):
        make = row['make']
        model = row['model']
        return 1 if make in suv_truck_models and model in suv_truck_models[make] else 0
    df['is_suv_truck'] = df.apply(is_suv_or_truck, axis=1)
    
    # Reliability indicator
    def is_reliable(row):
        make = row['make']
        model = row['model']
        return 1 if make in reliable_models and model in reliable_models[make] else 0
    df['is_reliable'] = df.apply(is_reliable, axis=1)
    
    # Country of origin
    df['country_of_origin'] = df['make'].map(country_of_origin)
    
    # Popularity (frequency count of each make and model)
    popularity = df.groupby(['make', 'model']).size().reset_index(name='popularity')
    df = df.merge(popularity, on=['make', 'model'], how='left')
    
    return df