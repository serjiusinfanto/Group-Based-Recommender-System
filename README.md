# Restaurant Recommendation System

## Overview
This project develops a machine-learning-based restaurant recommendation system to suggest dining options based on user preferences and restaurant attributes. The hybrid model leverages collaborative filtering, content-based filtering, and neural networks to deliver personalized and group-based recommendations.

## Features
- **Collaborative Filtering**: Utilizes user-review interactions to recommend restaurants based on similar users/items.
- **Content-Based Filtering**: Matches user preferences with restaurant attributes (e.g., ambiance, price range, cuisine).
- **Hybrid Model**: Combines collaborative and content-based methods using neural networks to enhance prediction accuracy.
- **Real-Time Restaurant Status**: Integrates Google Places API to fetch open/closed statuses for restaurants.

## Dataset
- **Source**: Yelp Open Dataset
- **Files Used**:
  - `business.json`: Contains restaurant attributes and metadata.
  - `review.json`: Contains user reviews and ratings.

## Key Technologies
- **Programming Languages**: Python
- **Libraries**:
  - Pandas, NumPy, Scikit-learn, PyTorch, Dask, Matplotlib, Plotly
- **External APIs**: Google Places API for real-time status
- **Preprocessing**:
  - JSON parsing
  - One-hot encoding
  - KNN imputation
  - Feature scaling
- **Evaluation Metrics**:
  - Root Mean Square Error (RMSE)
  - Mean Average Precision (MAP)
  - Precision@K

## Results
- **Hybrid Model Performance**:
  - Train RMSE: 1.19
  - Test RMSE: 1.16
  - MAP: 0.77
  - Precision@5: 0.75

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/serjiusinfanto/Group-Based-Recommender-System.git
