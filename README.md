# 🎬 Movie Recommendation System (MovieLens 100K)

A collaborative filtering-based movie recommendation system built on the classic MovieLens 100K dataset. The system implements and compares multiple recommendation approaches — memory-based (User-User and Item-Item CF) and model-based (SVD matrix factorization) — along with cold-start handling and rigorous evaluation.

---

## 📌 Project Highlights

- Implemented **3 recommendation strategies**: Item-Item CF, User-User CF, and SVD (Matrix Factorization)
- Compared **Cosine Similarity vs. Pearson Correlation** for memory-based approaches
- Applied **Bayesian weighted popularity scoring** to handle rating bias
- Addressed the **Cold Start Problem** with a popularity-based fallback
- Evaluated using **RMSE, MAE, Precision@K, and Recall@K**
- Used **KneeLocator** for data-driven filtering thresholds
- Performed **Grid Search hyperparameter tuning** with statistical significance testing (paired t-test)

---

## 🏆 STAR Format — Project Summary

### Situation
Movie streaming platforms face a core challenge: with thousands of titles available, users struggle to discover content they'll enjoy. The MovieLens 100K dataset (100,000 ratings by 943 users across 1,682 movies) provides a benchmark environment to build and evaluate recommendation algorithms. The matrix is **93.7% sparse** — meaning most user-movie pairs have no rating — which makes this a non-trivial prediction problem.

### Task
Design and implement a complete recommendation pipeline that:
- Handles missing/sparse data appropriately
- Generates meaningful personalized recommendations
- Evaluates multiple approaches objectively
- Gracefully handles edge cases (new users, unpopular items)

### Action
- **Data Loading & Cleaning**: Ingested three separate data files (ratings, users, items), handled null values, dropped uninformative columns (`video_release_date`), and merged into a unified DataFrame.
- **Exploratory Data Analysis**: Analysed rating distributions, user activity patterns, genre frequencies, and demographic breakdowns. Used Bayesian averaging to build a noise-resistant popularity score.
- **User-Item Matrix Construction**: Built a pivot table (943×1682) and computed matrix density (6.3%), confirming the need for matrix factorization.
- **Memory-Based CF**: Built Item-Item and User-User collaborative filters using both Cosine Similarity and Pearson Correlation. Applied mean-centering and minimum co-rater thresholds to improve similarity quality.
- **Model-Based CF (SVD)**: Used the `Surprise` library's SVD implementation. Filtered data using KneeLocator-derived thresholds to remove noise from very inactive users and unpopular movies.
- **Evaluation**: Computed RMSE, MAE, Precision@10, and Recall@10. Ran GridSearchCV over 27 hyperparameter combinations and validated improvements using a paired t-test.
- **Cold Start Handling**: Implemented a fallback mechanism that returns Bayesian-popularity-ranked movies for new/unknown users.

### Result
- **Baseline SVD RMSE: 0.8535 | MAE: 0.6575** on a 1–5 scale
- **Precision@10: 0.9270** — 92.7% of top-10 recommendations were relevant (rated ≥ 4.0)
- **Recall@10: ~0.57** — the model surfaces ~57% of all movies a user would have liked
- Hyperparameter tuning did not yield a statistically significant improvement (p = 0.22), validating the baseline as the optimal configuration
- Cold-start fallback ensures 100% coverage — the system always returns recommendations

---

## 🗂️ Dataset

**MovieLens 100K** — [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)

| File | Description |
|---|---|
| `u.data` | 100,000 ratings (user_id, movie_id, rating, timestamp) |
| `u.user` | 943 users with age, gender, occupation, zip code |
| `u.item` | 1,682 movies with title, release date, and 19 genre flags |

- Rating scale: 1–5 stars
- Matrix sparsity: **93.7%**
- Average rating: **3.53**

---

## 🧪 Methodology

### 1. Data Preprocessing
- Converted Unix timestamps to datetime
- Dropped `video_release_date` (entirely null)
- Removed corrupted movie entry (movie_id 267)
- Filled missing IMDb URLs with a placeholder
- Merged all three tables into a single analytical DataFrame

### 2. Exploratory Data Analysis (EDA)
- **Rating Distribution**: Peaks at 3–4 stars; moderate positive skew
- **User Activity**: Top user rated 737 movies; high variance in activity levels
- **Genre Analysis**: Drama (43.1%), Comedy (30%), Thriller (16.1%) are most frequent
- **Sparsity Confirmation**: Only 6.3% of the user-item matrix is populated
- **KneeLocator Threshold**: Data-driven cutoffs for minimum ratings per user and per movie, reducing noise without over-filtering
- **Bayesian Popularity Score**: Weighted average formula `(v/(v+m)) * R + (m/(v+m)) * C` shrinks low-vote movie scores toward the global mean

### 3. Memory-Based Collaborative Filtering

#### Item-Item CF
- Built item similarity matrix via Cosine Similarity (mean-centered) and Pearson Correlation
- Added minimum co-rater threshold (15 users) to prevent spurious high similarities
- Recommendations based on movies most similar to a seed movie

#### User-User CF
- Built user similarity matrix with row-wise mean-centering (reduces generous/strict rater bias)
- Used **Baseline Estimate formula** for prediction:
  `pred(u,i) = mean(u) + Σ[s(u,v) * (r(v,i) - mean(v))] / Σ|s(u,v)|`
- Predictions can exceed the 1–5 scale (expected behavior of deviation-based formulas)

### 4. Model-Based CF — SVD (Surprise Library)
- Applied KneeLocator-filtered dataset as input
- Default SVD: `n_factors=100`, evaluated via 80/20 train-test split
- GridSearchCV over 27 combinations (`n_factors`, `n_epochs`, `lr_all`, `reg_all`)
- Validated with paired t-test — no statistically significant improvement from tuning
- Retrained baseline on full dataset for final recommendations

### 5. Cold Start Handling
- New users (not in training data) receive **top-N popular movies** based on Bayesian popularity score
- Ensures zero silent failures — the system always returns a result

---

## 📊 Evaluation Metrics

| Metric | Value |
|---|---|
| RMSE (Baseline SVD) | 0.8535 |
| MAE (Baseline SVD) | 0.6575 |
| Precision@10 | 0.9270 |
| Recall@10 | ~0.57 |
| Tuned RMSE | ~0.851 (not statistically better) |

> Relevant threshold for Precision/Recall: ratings ≥ 4.0

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | pandas, NumPy |
| ML / Recommendation | Surprise (SVD, NMF, KNNBasic), scikit-learn (cosine_similarity, TruncatedSVD) |
| Statistical Testing | SciPy (ttest_rel) |
| Visualization | Matplotlib, Seaborn |
| Utilities | kneed (KneeLocator), collections (defaultdict) |
| Environment | Kaggle Notebooks |

---

## 📁 Project Structure

```
movie-recommender/
│
├── movie-recommender.ipynb     # Main notebook (EDA + Modeling + Evaluation)
├── insights_summary.json       # Dataset-level statistics
├── popularity_scores.csv       # Bayesian popularity scores per movie
├── top_popular_movies.csv      # Top 10 movies by weighted score
├── movie_rating_stats.csv      # Per-movie avg rating and count
├── rating_counts.csv           # Distribution of star ratings
├── gender_avg_rating.csv       # Mean rating by gender
├── occupation_counts.csv       # User count by occupation
└── README.md
```

---

## 🔑 Key Design Decisions & Learnings

- **Why SVD over KNN?** 93.7% sparsity makes neighbourhood-based methods unreliable — SVD handles this via latent factor decomposition
- **Why Bayesian scoring?** Raw average ratings for low-count movies are noisy; Bayesian shrinkage produces more reliable rankings
- **Why KneeLocator for thresholds?** Manual threshold selection is arbitrary; the elbow in the rating-count curve provides a principled, data-driven cutoff
- **Why Pearson over Cosine?** Pearson corrects for user-level rating scale bias (harsh vs. generous raters), making similarity more meaningful
- **Why retrain on full data?** After evaluation, retraining on 100% of available ratings reduces prediction variance for production use

---

## 🚧 Known Limitations, Pitfalls & Future Work

- **Data Leakage in GridSearchCV**: Hyperparameter tuning was performed on the full `data` object rather than being isolated strictly to the training split. This means the test set leaked into the cross-validation folds, making the "Tuned RMSE" slightly over-optimistic.
- **Global Thresholds & Popularity**: Bayesian popularity scores and KneeLocator thresholds were computed on the entire dataset prior to splitting. In a strict ML pipeline, these statistics should only be computed on the training set to prevent target leakage.
- **Temporal Leakage**: A random train/test split was used. In real-world recommender systems, user preferences drift over time. A time-based split (e.g., using the last 20% of chronological ratings as the test set) would be a more robust evaluation method.
- **Ratings above 5.0**: The deviation-based prediction formula can produce out-of-range scores; clipping is applied in the final function.
- **No content-based features used**: Genre, release year, and occupation data are explored in EDA but not integrated into the model.
- **Hybrid Model**: Combining SVD scores with popularity scores using a tunable α weight is a logical next step.

---

## 🚀 How to Run

1. Clone/download the repository
2. Open `movie-recommender.ipynb` in Kaggle or Jupyter
3. Ensure the MovieLens 100K dataset is available at `/kaggle/input/movie-lens/ml-100k/`
4. Run all cells in order

```bash
pip install scikit-surprise kneed
```

---

## 👤 Author

**Aryan**
Aspiring Data Scientist | Building end-to-end ML projects to transition into industry

---

## 📄 License

This project uses the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) provided by GroupLens Research. Dataset is for non-commercial research/educational use.
