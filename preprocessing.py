import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.impute import KNNImputer

#Preprocessing only the business csv file as the rest of the files are already clean
def preprocess_business_file(business_path):
    business_df = pd.read_csv(business_path, low_memory=False)
    business_df = business_df.drop(columns=['categories'])


    geolocator = Nominatim(user_agent="restaurant_geocoder")
    #Function to fill missing address values using latitude and longitude from geolocator library
    def reverse_geocode(lat, lon):
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location:
                address = location.raw['address']
                return address.get('road', '')
            return ''  # If geocoding fails, return empty strings
        except Exception as e:
            print(f"Error with reverse geocoding: {e}")
            return ''
    
    #Dealing with null values for address column
    business_df['address'] = business_df.apply(lambda row : reverse_geocode(row['latitude'],row['longitude']) if pd.isnull(row['address']) else row['address'],axis=1)
    
    #Dealing with null values for good_for_groups column
    business_df['good_for_groups'] = business_df['good_for_groups'].astype('boolean')
    business_df['good_for_groups'] = business_df['good_for_groups'].fillna( business_df['good_for_groups'].mode()[0])
    
    #Dealing with null values and one hot encoding for price_range column
    knn_imputer = KNNImputer()
    business_df['price_range'] = business_df['price_range'].astype('Int64')
    business_df[['price_range']] = knn_imputer.fit_transform(business_df[['price_range']])
    business_df['price_range'] = business_df['price_range'].round().astype(int)
    
    #Dealing with null values for take out column
    business_df['take_out'] = business_df['take_out'].astype('boolean')
    business_df['take_out'] = business_df['take_out'].fillna( business_df['take_out'].mode()[0])

    #Encoding true and false values for multiple columns
    columns_to_be_mapped = ['good_for_groups','take_out','touristy_ambience','hipster_ambience','romantic_ambience','divey_ambience','intimate_ambience','trendy_ambience','upscale_ambience','classy_ambience','casual_ambience']
    for col in columns_to_be_mapped:
        business_df[col] = business_df[col].map({
            True : 1,
            False : 0
        })
    
    #Dealing with null values and one hot encoding for alcohol column
    business_df['alcohol'] = business_df['alcohol'].fillna( business_df['alcohol'].mode()[0])
    business_df['alcohol'] = business_df['alcohol'].replace('False', 'no_alcohol')
    alcohol_one_hot = pd.get_dummies(business_df['alcohol'], prefix='alcohol')
    business_df = pd.concat([business_df, alcohol_one_hot], axis=1)
    business_df = business_df.drop(columns='alcohol')
    print(business_df.duplicated(subset='business_id').sum())
    return business_df

if __name__ == "__main__":
    business_path = 'data/business_data.csv'

    business_df = preprocess_business_file(business_path)
    business_df.to_csv(r'data/business_data_cleaned.csv',index=False)

