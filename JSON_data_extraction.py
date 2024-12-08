import pandas as pd
import numpy as np
import json

class YelpDataProcessor:

    def __init__(self,business_json_path,checkin_json_path,review_json_path,user_json_path):
        self.business_json_path = business_json_path
        self.checkin_json_path = checkin_json_path
        self.review_json_path = review_json_path
        self.user_json_path = user_json_path
        self.business_df = None
        self.checkin_df = None
        self.review_df = None
        self.user_df = None
    
    #Function to extract business json file
    def extract_business_data(self):
        business_data = []
        with open(self.business_json_path,'r',encoding='utf-8') as json_file:
            for data in json_file:
                business_data.append(json.loads(data))
        business_df = pd.DataFrame(business_data)
        

        #Filtering out business that are only restaurants and that are currently open
        business_df = business_df[(business_df['is_open'] == 1) & (business_df['categories'].str.contains('Restaurants'))]

        #Filtering out the essential features
        business_df = business_df[['business_id','name','address','city','state','postal_code','latitude','longitude','stars','review_count','is_open','attributes','categories']]
        
        business_df.rename(columns={'stars':'business_rating'},inplace=True)
        business_df.rename(columns={'review_count':'business_review_count'},inplace=True)
        

        #Extract attributes
        def extract_attributes(attributes):
            
            if attributes is None:
                return {
                    'good_for_groups' : None,
                    'price_range' : None,
                    'take_out' : None,
                    'touristy_ambience' : None,
                    'hipster_ambience' : None,
                    'romantic_ambience' : None,
                    'divey_ambience' : None,
                    'intimate_ambience' : None,
                    'trendy_ambience' : None,
                    'upscale_ambience' : None,
                    'classy_ambience' : None,
                    'casual_ambience' : None,
                    'alcohol' : None
                }
            
            ambience = attributes.get('Ambience',{})
            
            if isinstance(ambience,str):
                try:
                    ambience = ambience.replace("'","\"")
                    ambience = ambience.strip('{}')
                    ambience = ambience.split(', ')
                    ambience = { key.strip('"'): value.strip().lower() == 'true' for key, value in (pair.split(': ') for pair in ambience)}
                except (KeyError,ValueError) as e:
                    print(f"Error parsing Ambience string {ambience} | Error {e}")
                    ambience = {}
                    
            def clean_alcohol(value):
                if isinstance(value,str):
                    value = value.strip("'u").strip("'")
                    if value == "none":
                        return False
                    return value
                return value
            
            return {
                    'good_for_groups' : attributes.get('RestaurantsGoodForGroups'),
                    'price_range' : attributes.get('RestaurantsPriceRange2'),
                    'take_out' : attributes.get('RestaurantsTakeOut'),
                    'touristy_ambience' : ambience.get('touristy'),
                    'hipster_ambience' : ambience.get('hipster'),
                    'romantic_ambience' : ambience.get('romantic'),
                    'divey_ambience' : ambience.get('divey'),
                    'intimate_ambience' : ambience.get('intimate'),
                    'trendy_ambience' : ambience.get('trendy'),
                    'upscale_ambience' : ambience.get('upscale'),
                    'classy_ambience' : ambience.get('classy'),
                    'casual_ambience' : ambience.get('casual'),
                    'alcohol' : clean_alcohol(attributes.get('Alcohol'))
                }
    
        
        #Applying extract attributes function to the attributes column and storing it as a separate dataframe
        attributes_df = business_df['attributes'].apply(extract_attributes).apply(pd.Series)
        attributes_df = attributes_df.fillna({
                    'take_out' : False,
                    'good_for_groups' : False,
                    'touristy_ambience' : False,
                    'hipster_ambience' : False,
                    'romantic_ambience' : False,
                    'divey_ambience' : False,
                    'intimate_ambience' : False,
                    'trendy_ambience' : False,
                    'upscale_ambience' : False,
                    'classy_ambience' : False,
                    'casual_ambience' : False,
                    'alcohol' : False
        })
        
        #One-hot encoding the categories column
        business_df['categories'] = business_df['categories'].apply(lambda x:x.split(', ') if isinstance(x,str) else [])
        categories_expanded = business_df['categories'].apply(pd.Series)
        categories_dummies = categories_expanded.stack().str.strip().str.get_dummies().groupby(level=0).sum()
        business_df = pd.concat([business_df,categories_dummies],axis=1)

        #Concatenating the attributes dataframe and dropping the old attributes column
        self.business_df = pd.concat([business_df.drop(columns=['attributes']),attributes_df],axis=1)
        print("Business DataFrame:\n")
        print(self.business_df.head())

        return self.business_df
    

    #Function to extract review json file
    def extract_review_data(self):
        review_data = []
        with open(self.review_json_path,'r',encoding='utf-8') as json_file:
            for data in json_file:
                review_data.append(json.loads(data))
        review_df = pd.DataFrame(review_data)
        review_df.rename(columns={'stars':'review_rating'},inplace=True)
        self.review_df = review_df[['review_id','user_id','business_id','review_rating']]
        print("Review DataFrame:\n")
        print(self.review_df.head())

        return self.review_df
    
    #Function to extract user json file
    def extract_user_data(self):
        user_data = []
        with open(self.user_json_path,'r',encoding='utf-8') as json_file:
            for data in json_file:
                user_data.append(json.loads(data))
        user_df = pd.DataFrame(user_data)
        user_df.rename(columns={'review_count':'user_review_count'},inplace=True)
        self.user_df = user_df[['user_id','user_review_count','average_stars']]
        print("User DataFrame:\n")
        print(self.user_df.head())

        return self.user_df
    
    #Function to extract checkin json file
    def extract_checkin_data(self):
        checkin_data = []
        with open(self.checkin_json_path,'r',encoding='utf-8') as json_file:
            for data in json_file:
                checkin_data.append(json.loads(data))
        checkin_df = pd.DataFrame(checkin_data)
        
        def calcualte_checkins(date):
            if pd.notna(date):
                return len(date.split(', '))
            return 0
        
        checkin_df['no_of_checkins'] = checkin_df['date'].apply(calcualte_checkins)
        self.checkin_df = checkin_df[['business_id','no_of_checkins']]
        print("Checkin DataFrame:\n")
        print(self.checkin_df.head())

        return self.checkin_df

        
    
    #Function to merge all the dataframes together
    def merge_data(self):
        #check if any dataframe is empty
        if any(df is None for df in [self.business_df,self.checkin_df,self.review_df,self.user_df]):
            raise ValueError("Extract all datasets before merging")
        print(f"Length of business data frame:{len(self.business_df)}")
        print(f"Length of review data frame:{len(self.review_df)}")
        merged_df = pd.merge(self.review_df,self.business_df,how="inner",on='business_id')
        print(f"Length of data after 1st merge:{len(merged_df)}")
        merged_df = pd.merge(merged_df,self.user_df,how="inner",on='user_id')
        print(f"Length of data after 2nd merge:{len(merged_df)}")
        merged_df = pd.merge(merged_df,self.checkin_df,how="left",on="business_id")
        

        merged_df['no_of_checkins'] = merged_df['no_of_checkins'].fillna(0)
        return merged_df
    

if __name__ == "__main__":
    business_path = "data/yelp_academic_dataset_business.json"
    checkin_path = "data/yelp_academic_dataset_checkin.json"
    review_path = "data/yelp_academic_dataset_review.json"
    user_path = "data/yelp_academic_dataset_user.json"

    json_processor = YelpDataProcessor(business_path,checkin_path,review_path,user_path)

    business_df = json_processor.extract_business_data()
    business_df.to_csv(r'data/business_data.csv',index=False)
    checkin_df = json_processor.extract_checkin_data()
    checkin_df.to_csv(r'data/checkin_data.csv',index=False)
    review_df = json_processor.extract_review_data()
    review_df.to_csv(r'data/review_data.csv',index=False)
    user_df = json_processor.extract_user_data()
    user_df.to_csv(r'data/user_data.csv',index=False)

    #final_data = json_processor.merge_data()

    

    



        



        


