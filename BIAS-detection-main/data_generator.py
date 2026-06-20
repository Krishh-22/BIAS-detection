import pandas as pd
import numpy as np

def generate_biased_data(n=1000):
    np.random.seed(42)
    
    # Features
    experience = np.random.randint(0, 21, n)
    test_score = np.random.randint(40, 101, n)
    gender = np.random.choice([0, 1], n)  # 0: Female, 1: Male
    
    # Bias Logic: Men get a "hidden" +15 point bonus in this historical data
    # Women must work harder to get 'hired' in the training set
    hiring_score = (experience * 3) + (test_score * 0.5) + (gender * 15) + np.random.normal(0, 5, n)
    hired = (hiring_score > 75).astype(int)
    
    df = pd.DataFrame({
        'experience': experience,
        'test_score': test_score,
        'gender': gender,
        'hired': hired
    })
    
    df.to_csv("recruitment_data.csv", index=False)
    print("✅ Professional Dataset 'recruitment_data.csv' generated successfully.")

if __name__ == "__main__":
    generate_biased_data()