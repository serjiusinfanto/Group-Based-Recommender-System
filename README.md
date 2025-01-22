# Restaurant Recommendation System

## Overview
This project focuses on developing a machine-learning-based recommendation system that predicts restaurants based on user preferences and restaurant attributes. The system employs collaborative filtering, content-based filtering, and a hybrid approach to provide personalized and group-based recommendations. Real-time restaurant information is also integrated through the Google Places API.

## Key Features
- **Collaborative Filtering**: Leverages user-review interactions to suggest restaurants based on similar users or items.
- **Content-Based Filtering**: Matches user preferences with restaurant features like price range, ambiance, and cuisine.
- **Hybrid Model**: Combines collaborative and content-based methods enhanced with a neural network for higher prediction accuracy.
- **Group Recommendations**: Aggregates user profiles to make recommendations tailored for groups.
- **Real-Time Information**: Fetches live restaurant operating status using the Google Places API.

## Dataset
- **Source**: [Yelp Dataset](https://www.yelp.com/dataset)
- **Files Used**:
  - `business.json`: Restaurant attributes and metadata for 34,987 restaurants.
  - `review.json`: User reviews and ratings totaling over 6.7 million data points.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**:
  - **Data Processing**: Pandas, NumPy
  - **Machine Learning**: Scikit-learn, PyTorch, Scipy
  - **Visualization**: Matplotlib, Plotly
  - **Data Scaling**: Dask for handling large-scale data
  - **API Integration**: Google Places API for real-time updates
- **Preprocessing**:
  - JSON parsing
  - KNN Imputation for handling missing values
  - One-hot encoding and feature scaling

## Model Evaluation Metrics
- **Root Mean Square Error (RMSE)**:
  - Collaborative Filtering: Test RMSE = 4.01
  - Content-Based Filtering: Test RMSE = 3.21
  - Hybrid Model: Test RMSE = 1.16
- **Mean Average Precision (MAP)**:
  - Hybrid Model: 0.77
- **Precision@K (K=5)**:
  - Collaborative Filtering: 1.00
  - Hybrid Model: 0.75

## Preprocessing Workflow
1. **Data Cleaning**:
   - Filtered for open restaurants and essential features.
   - Handled missing values using KNN imputation for critical attributes like price range.
2. **Encoding**:
   - Converted categorical attributes into numerical values using one-hot encoding.
3. **Feature Scaling**:
   - Normalized features with wide value ranges to avoid disproportionate model impact.
4. **Data Reduction**:
   - Reduced the dataset to 500,000 users and 50,000 businesses for computational efficiency.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/serjiusinfanto/Group-Based-Recommender-System.git
  

## Results
* The Hybrid Model achieved the best performance, combining collaborative and content-based features with a neural network.
* Enhanced prediction accuracy and improved ranking metrics, making it suitable for real-world applications.

## Future Enhancements
* Integration of user sentiment analysis from reviews for refined recommendations.
* Dynamic updates for real-time restaurant availability and wait times.
* Adopting reinforcement learning to adapt recommendations based on user interactions over time.
