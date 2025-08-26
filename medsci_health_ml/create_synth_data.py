import pandas as pd
import numpy as np
import os

# Ensure datasets folder exists
os.makedirs("datasets", exist_ok=True)

# Increase dataset size to 2000 rows for complexity
n_rows = 2000
np.random.seed(42)

# 1. Simulated Diabetes Dataset (like Pima, with extra features)
diabetes = pd.DataFrame({
    "age": np.random.randint(20, 80, n_rows),
    "bmi": np.random.uniform(18, 40, n_rows),
    "glucose": np.random.uniform(70, 200, n_rows),
    "blood_pressure": np.random.uniform(60, 120, n_rows),
    "insulin": np.random.uniform(15, 300, n_rows),
    "skin_thickness": np.random.uniform(10, 50, n_rows),
    "diabetes_pedigree": np.random.uniform(0.1, 2.5, n_rows),
    "pregnancies": np.random.randint(0, 15, n_rows),
    "diabetes": np.random.randint(0, 2, n_rows)
})
diabetes.to_csv("datasets/diabetes.csv", index=False)

# 2. Simulated Heart Disease Dataset (more features)
heart = pd.DataFrame({
    "age": np.random.randint(30, 80, n_rows),
    "sex": np.random.randint(0, 2, n_rows),
    "cholesterol": np.random.uniform(150, 300, n_rows),
    "resting_bp": np.random.uniform(80, 180, n_rows),
    "max_hr": np.random.uniform(90, 200, n_rows),
    "exercise_angina": np.random.randint(0, 2, n_rows),
    "st_depression": np.random.uniform(0, 6, n_rows),
    "slope": np.random.choice([1, 2, 3], n_rows),
    "ca": np.random.randint(0, 4, n_rows),
    "thal": np.random.choice([3, 6, 7], n_rows),
    "heart_disease": np.random.randint(0, 2, n_rows)
})
heart.to_csv("datasets/heart.csv", index=False)

# 3. Simulated Cancer Dataset (more features)
cancer = pd.DataFrame({
    "tumor_size": np.random.uniform(1, 10, n_rows),
    "mean_radius": np.random.uniform(5, 30, n_rows),
    "mean_texture": np.random.uniform(10, 40, n_rows),
    "mean_smoothness": np.random.uniform(0.05, 0.2, n_rows),
    "mean_symmetry": np.random.uniform(0.1, 0.4, n_rows),
    "fractal_dimension": np.random.uniform(0.05, 0.2, n_rows),
    "mean_compactness": np.random.uniform(0.02, 0.5, n_rows),
    "concavity": np.random.uniform(0, 0.5, n_rows),
    "concave_points": np.random.uniform(0, 0.3, n_rows),
    "malignant": np.random.randint(0, 2, n_rows)
})
cancer.to_csv("datasets/cancer.csv", index=False)

# Return summary (shapes and first few columns)
{
    "diabetes.csv": {"shape": diabetes.shape, "columns": diabetes.columns.tolist()[:6]},
    "heart.csv": {"shape": heart.shape, "columns": heart.columns.tolist()[:6]},
    "cancer.csv": {"shape": cancer.shape, "columns": cancer.columns.tolist()[:6]}
}
