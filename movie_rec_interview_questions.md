# Movie Recommendation Systems: Interview Questions & Answers

This document serves as a comprehensive reference for preparing for Machine Learning Engineer and Data Scientist interviews focusing on Recommendation Systems, Matrix Factorization, and Collaborative Filtering.

---

## 1. General Recommendation Systems

### Q: What is the difference between Content-Based Filtering and Collaborative Filtering? Which one is better?
**A:**
*   **Content-Based Filtering (CBF):** Recommends items by comparing the features of an item (e.g., genre, director, actors for movies) to the profile of the user's preferences.
    *   *Pros:* Doesn't suffer from the cold-start problem for *new items*. Highly transparent (you know exactly why a recommendation was made).
    *   *Cons:* Suffers from "filter bubble" or lack of serendipity. It only recommends things highly similar to what the user already consumed.
*   **Collaborative Filtering (CF):** Recommends items based on the past behavior and similarity of users or items (e.g., "Users who liked what you liked also liked...").
    *   *Pros:* Can recommend completely novel items (serendipity). Doesn't require manual feature engineering of items.
    *   *Cons:* Suffers heavily from the cold-start problem (for both users and items) and sparsity issues.
*   *Which is better?* Neither is strictly better. Modern industry standards almost exclusively use **Hybrid Systems** that combine both approaches to leverage the strengths of each.

### Q: Explain the "Cold Start" problem and how you would address it in a production environment.
**A:**
The cold start problem occurs when the system cannot draw any inferences for users or items because it has not yet gathered sufficient information (ratings/interactions).
*   **New User:** We don't know what they like.
    *   *Solution:* Popularity-based fallbacks (recommend global top charts), geographic trends, or explicit onboarding (asking them to pick 3 genres or rate 5 items when they sign up).
*   **New Item:** Nobody has interacted with it yet.
    *   *Solution:* Content-based filtering (using its metadata like tags or descriptions) to recommend it to users who like similar metadata, or dedicating a small percentage of recommendation slots to "exploration" (A/B testing new items).
*   **New System:** Extreme sparsity.
    *   *Solution:* Use implicit feedback (clicks, watch time) instead of waiting for explicit ratings, or use third-party data.

---

## 2. Matrix Factorization & SVD

### Q: Explain the intuition behind Matrix Factorization and "Latent Features".
**A:**
Matrix Factorization assumes that the sparse User-Item interaction matrix can be approximated by the dot product of two lower-dimensional matrices: a User matrix and an Item matrix. 
**Latent Features** are the hidden dimensions (columns) in these smaller matrices. They represent abstract concepts that the model *discovers on its own*. 
*   *Intuition:* For movies, a latent feature might map to "amount of action," "quirkiness," or "directed by Christopher Nolan." The model doesn't explicitly know these labels, but it learns that certain users consistently rate movies high if those movies share this abstract numerical feature.

### Q: Standard Singular Value Decomposition (SVD) cannot handle missing values (NaNs). How do algorithms like Surprise's SVD or Spark's ALS solve this?
**A:**
Standard mathematical SVD requires a dense, complete matrix. If we impute all missing ratings with 0s or averages, it heavily distorts the data and is computationally explosive.
*   Instead of solving the exact linear algebra SVD, recommendation algorithms like FunkSVD (used in Surprise) or ALS (Alternating Least Squares) formulate it as an **optimization problem**. 
*   They only calculate the error (loss) on the *known* ratings. They use Gradient Descent (or ALS) to iteratively update the User and Item matrices to minimize the Root Mean Squared Error (RMSE) of the *observed* values, while applying regularization to prevent overfitting.

---

## 3. Collaborative Filtering Details

### Q: Why is Item-Item Collaborative Filtering generally preferred over User-User in large-scale production systems (like Amazon)?
**A:**
*   **Stability:** Item profiles (how an item relates to other items) change much more slowly than user profiles. A user's taste can pivot quickly, but the similarity between *Star Wars* and *Star Trek* is relatively static.
*   **Scalability / Compute:** In most consumer systems, the number of users is vastly larger than the number of items ($U \gg I$). Calculating an $I \times I$ similarity matrix is far less computationally intensive and requires less memory than a $U \times U$ matrix.
*   **Caching:** Because the Item-Item matrix is smaller and more stable, it can be computed offline (batch processing) and easily cached in memory for fast real-time serving.

### Q: When calculating User-User or Item-Item similarity, why might we prefer Pearson Correlation over Cosine Similarity?
**A:**
Cosine similarity measures the angle between vectors but ignores magnitude. Pearson Correlation solves the "tough grader / generous grader" bias by mean-centering the data. It subtracts a user's average rating from all their ratings before calculating similarity. This ensures that a user who rates everything 4-5 stars can still be matched accurately with a user who rates everything 1-2 stars, provided their *relative* preferences match.

---

## 4. Evaluation Metrics

### Q: You trained a model with a low RMSE, but user engagement dropped in A/B testing. What might be wrong?
**A:**
RMSE evaluates the average accuracy of predicted ratings across *all* items. However, users only ever see the Top-10 or Top-20 recommendations. 
*   A model might be very good at predicting that a user will hate a movie (predicting a 1.2 instead of a 1.0), which improves RMSE but adds zero business value.
*   RMSE does not penalize ranking errors at the top of the list heavily enough.
*   *Solution:* We should rely more on ranking metrics like **Precision@K**, **Recall@K**, **MAP@K** (Mean Average Precision), or **NDCG** (Normalized Discounted Cumulative Gain) which evaluate the quality and ranking of the *actual items shown to the user*.

### Q: What is the "Sparsity Problem" and why is it detrimental to Memory-Based (KNN) models?
**A:**
Sparsity refers to the percentage of missing values in the user-item matrix (often >95% in real systems). Memory-based models rely on calculating overlaps (co-ratings) to find similarities. If two users have very few ratings, the chance of them having rated the *same* items is extremely low. Thus, the model cannot confidently calculate similarity or make reliable recommendations. Matrix Factorization mitigates this by inferring relationships through latent factors rather than requiring direct co-ratings.

---

## 5. System Design & Production

### Q: How do you prevent popularity bias (where blockbuster movies drown out niche content)?
**A:**
Popularity bias hurts the discovery of niche content (the "long tail"). Mitigation strategies include:
1.  **Bayesian Average / Weighted Ratings:** Penalizing items with very few ratings, but also normalizing massive blockbusters.
2.  **Inverse User Frequency (IUF):** Similar to TF-IDF in NLP, we can down-weight the importance of co-rating an extremely popular movie (since everyone watched it) and up-weight co-rating a rare niche movie (which indicates a strong specific taste overlap).
3.  **Exploration vs. Exploitation:** Dedicate a portion of the recommendation carousel (e.g., 10%) specifically to less popular or newer items.

### Q: If a user rates a movie right now, how does the system immediately update their recommendations?
**A:**
Re-training an SVD or matrix factorization model on millions of users takes hours (done in batch processing). To update in real-time (online serving):
*   **Item-Item CF:** Since the $I \times I$ similarity matrix is pre-computed and cached, when a user rates Item A, the system immediately looks up the most similar items to A from the cache and adjusts the user's real-time recommendations.
*   **Folding-in (SVD):** Keep the Item latent matrix fixed. Use the user's new ratings and the fixed Item matrix to do a quick localized gradient descent or least-squares update to update *only* that specific User's latent vector.
