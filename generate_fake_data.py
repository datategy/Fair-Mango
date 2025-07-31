#!/usr/bin/env python3
"""
Fake Data Generator for Fair-Mango Interface Testing

This script generates multiple diverse datasets to test the Fair-Mango interface
with different combinations of sensitive attributes, positive outcomes, and data patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def create_heart_disease_dataset(n_samples: int = 918) -> pd.DataFrame:
    """
    Create a synthetic heart disease dataset similar to the one expected by tests.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    pd.DataFrame
        Generated heart disease dataset
    """
    # Generate basic demographics
    sex = np.random.choice(
        ["M", "F"], n_samples, p=[0.79, 0.21]
    )  # Male-heavy as in real data
    age = np.random.randint(28, 78, n_samples)

    # Chest pain types
    chest_pain_types = ["ASY", "NAP", "ATA", "TA"]
    chest_pain_weights = [0.5, 0.2, 0.15, 0.15]  # ASY is most common
    chest_pain = np.random.choice(chest_pain_types, n_samples, p=chest_pain_weights)

    # Other medical indicators
    resting_bp = np.random.normal(130, 20, n_samples).astype(int)
    resting_bp = np.clip(resting_bp, 80, 200)

    cholesterol = np.random.normal(240, 50, n_samples).astype(int)
    cholesterol = np.clip(cholesterol, 120, 400)

    fasting_bs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

    resting_ecg = np.random.choice(
        ["Normal", "ST", "LVH"], n_samples, p=[0.6, 0.2, 0.2]
    )

    max_hr = np.random.normal(150, 25, n_samples).astype(int)
    max_hr = np.clip(max_hr, 60, 202)

    exercise_angina = np.random.choice(["N", "Y"], n_samples, p=[0.6, 0.4])

    oldpeak = np.random.exponential(1.0, n_samples)
    oldpeak = np.round(oldpeak, 1)

    st_slope = np.random.choice(["Up", "Flat", "Down"], n_samples, p=[0.5, 0.4, 0.1])

    # Generate heart disease with realistic correlations
    # Higher probability for: older age, male, ASY chest pain, exercise angina
    heart_disease_prob = 0.3  # Base probability

    # Adjust probability based on features
    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (sex == "M") * 0.2  # Males more likely
    prob_adjustments += (age > 55) * 0.15  # Older age
    prob_adjustments += (chest_pain == "ASY") * 0.25  # Asymptomatic chest pain
    prob_adjustments += (exercise_angina == "Y") * 0.2  # Exercise angina
    prob_adjustments += (cholesterol > 250) * 0.1  # High cholesterol

    final_probs = np.clip(heart_disease_prob + prob_adjustments, 0.05, 0.95)
    heart_disease = np.random.binomial(1, final_probs)

    # Generate predictions with some noise (85% accuracy)
    prediction_accuracy = 0.85
    heart_disease_pred = heart_disease.copy()

    # Add prediction errors
    error_indices = np.random.choice(
        n_samples, int(n_samples * (1 - prediction_accuracy)), replace=False
    )
    heart_disease_pred[error_indices] = 1 - heart_disease_pred[error_indices]

    # Generate exercise angina predictions
    exercise_angina_binary = (exercise_angina == "Y").astype(int)
    exercise_angina_pred = exercise_angina_binary.copy()

    # Add some prediction errors for exercise angina
    error_indices_ea = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    exercise_angina_pred[error_indices_ea] = 1 - exercise_angina_pred[error_indices_ea]
    exercise_angina_pred = np.where(exercise_angina_pred == 1, "Y", "N")

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
            "HeartDisease": heart_disease,
            "HeartDiseasePred": heart_disease_pred,
            "ExerciseAnginaPred": exercise_angina_pred,
        }
    )

    return df


def create_hiring_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create a synthetic hiring dataset for fairness testing.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    pd.DataFrame
        Generated hiring dataset
    """
    # Demographics
    gender = np.random.choice(["Male", "Female"], n_samples, p=[0.6, 0.4])
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        n_samples,
        p=[0.6, 0.15, 0.15, 0.08, 0.02],
    )
    age_group = np.random.choice(
        ["18-30", "31-45", "46-60", "60+"], n_samples, p=[0.3, 0.4, 0.25, 0.05]
    )

    # Education
    education = np.random.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        n_samples,
        p=[0.2, 0.5, 0.25, 0.05],
    )

    # Experience (years)
    experience = np.random.exponential(5, n_samples).astype(int)
    experience = np.clip(experience, 0, 30)

    # Skills score (0-100)
    skills_score = np.random.normal(70, 15, n_samples)
    skills_score = np.clip(skills_score, 0, 100)

    # Interview score (0-100)
    interview_score = np.random.normal(75, 12, n_samples)
    interview_score = np.clip(interview_score, 0, 100)

    # Generate hiring decision with bias
    base_prob = 0.3

    # Add bias and legitimate factors
    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (gender == "Male") * 0.1  # Gender bias
    prob_adjustments += (race == "White") * 0.08  # Racial bias
    prob_adjustments += (education == "PhD") * 0.2  # Education matters
    prob_adjustments += (education == "Master") * 0.15
    prob_adjustments += (education == "Bachelor") * 0.1
    prob_adjustments += (experience > 5) * 0.15  # Experience matters
    prob_adjustments += (skills_score > 80) * 0.2  # Skills matter
    prob_adjustments += (interview_score > 80) * 0.15  # Interview performance

    final_probs = np.clip(base_prob + prob_adjustments, 0.05, 0.95)
    hired = np.random.binomial(1, final_probs)

    # Generate predictions (HR algorithm)
    hired_pred = hired.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    hired_pred[error_indices] = 1 - hired_pred[error_indices]

    df = pd.DataFrame(
        {
            "Gender": gender,
            "Race": race,
            "AgeGroup": age_group,
            "Education": education,
            "Experience": experience,
            "SkillsScore": skills_score.round(1),
            "InterviewScore": interview_score.round(1),
            "Hired": hired,
            "HiredPred": hired_pred,
        }
    )

    return df


def create_loan_approval_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """
    Create a synthetic loan approval dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    pd.DataFrame
        Generated loan approval dataset
    """
    # Demographics
    gender = np.random.choice(["M", "F"], n_samples, p=[0.55, 0.45])
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Asian"], n_samples, p=[0.65, 0.15, 0.12, 0.08]
    )
    age = np.random.randint(18, 75, n_samples)

    # Financial information
    income = np.random.lognormal(10.5, 0.8, n_samples).astype(int)
    income = np.clip(income, 20000, 500000)

    credit_score = np.random.normal(650, 100, n_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)

    loan_amount = np.random.lognormal(10.8, 0.6, n_samples).astype(int)
    loan_amount = np.clip(loan_amount, 5000, 1000000)

    debt_to_income = (
        np.random.beta(2, 5, n_samples) * 0.8
    )  # Most people have reasonable DTI

    employment_length = np.random.exponential(3, n_samples)
    employment_length = np.clip(employment_length, 0, 20)

    # Generate approval with realistic factors and some bias
    base_prob = 0.4

    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (credit_score > 700) * 0.3
    prob_adjustments += (credit_score > 600) * 0.15
    prob_adjustments += (income > 75000) * 0.2
    prob_adjustments += (debt_to_income < 0.3) * 0.15
    prob_adjustments += (employment_length > 2) * 0.1
    prob_adjustments -= (debt_to_income > 0.5) * 0.25
    prob_adjustments += (race == "White") * 0.05  # Subtle bias
    prob_adjustments += (gender == "M") * 0.03  # Subtle bias

    final_probs = np.clip(base_prob + prob_adjustments, 0.05, 0.95)
    approved = np.random.binomial(1, final_probs)

    # Generate predictions
    approved_pred = approved.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.12), replace=False)
    approved_pred[error_indices] = 1 - approved_pred[error_indices]

    df = pd.DataFrame(
        {
            "Gender": gender,
            "Race": race,
            "Age": age,
            "Income": income,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "DebtToIncome": debt_to_income.round(3),
            "EmploymentLength": employment_length.round(1),
            "Approved": approved,
            "ApprovedPred": approved_pred,
        }
    )

    return df


def create_criminal_justice_dataset(n_samples: int = 800) -> pd.DataFrame:
    """
    Create a synthetic criminal justice dataset for recidivism prediction.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    pd.DataFrame
        Generated criminal justice dataset
    """
    # Demographics
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Other"], n_samples, p=[0.5, 0.3, 0.15, 0.05]
    )
    gender = np.random.choice(["Male", "Female"], n_samples, p=[0.75, 0.25])
    age = np.random.randint(18, 65, n_samples)

    # Criminal history
    prior_convictions = np.random.poisson(2, n_samples)
    prior_convictions = np.clip(prior_convictions, 0, 15)

    crime_type = np.random.choice(
        ["Violent", "Property", "Drug", "Other"], n_samples, p=[0.25, 0.35, 0.25, 0.15]
    )

    sentence_length = np.random.exponential(2, n_samples)
    sentence_length = np.clip(sentence_length, 0.5, 20)

    # Social factors
    education_level = np.random.choice(
        ["Less than HS", "High School", "Some College", "College+"],
        n_samples,
        p=[0.4, 0.35, 0.2, 0.05],
    )

    employment_status = np.random.choice(
        ["Unemployed", "Part-time", "Full-time"], n_samples, p=[0.6, 0.25, 0.15]
    )

    # Generate recidivism with bias
    base_prob = 0.35

    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (race == "Black") * 0.1  # Systemic bias
    prob_adjustments += (gender == "Male") * 0.08
    prob_adjustments += (age < 25) * 0.15  # Younger offenders
    prob_adjustments += (prior_convictions > 3) * 0.2
    prob_adjustments += (crime_type == "Violent") * 0.1
    prob_adjustments += (employment_status == "Unemployed") * 0.12
    prob_adjustments -= (education_level == "College+") * 0.15
    prob_adjustments -= (age > 45) * 0.1

    final_probs = np.clip(base_prob + prob_adjustments, 0.05, 0.85)
    recidivism = np.random.binomial(1, final_probs)

    # Generate predictions (COMPAS-like algorithm)
    recidivism_pred = recidivism.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.18), replace=False)
    recidivism_pred[error_indices] = 1 - recidivism_pred[error_indices]

    df = pd.DataFrame(
        {
            "Race": race,
            "Gender": gender,
            "Age": age,
            "PriorConvictions": prior_convictions,
            "CrimeType": crime_type,
            "SentenceLength": sentence_length.round(1),
            "EducationLevel": education_level,
            "EmploymentStatus": employment_status,
            "Recidivism": recidivism,
            "RecidivismPred": recidivism_pred,
        }
    )

    return df


def create_college_admission_dataset(n_samples: int = 1200) -> pd.DataFrame:
    """
    Create a synthetic college admission dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate

    Returns
    -------
    pd.DataFrame
        Generated college admission dataset
    """
    # Demographics
    race = np.random.choice(
        ["White", "Asian", "Hispanic", "Black", "Other"],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.12, 0.03],
    )
    gender = np.random.choice(["Male", "Female"], n_samples, p=[0.48, 0.52])

    # Academic metrics
    gpa = np.random.normal(3.5, 0.4, n_samples)
    gpa = np.clip(gpa, 2.0, 4.0)

    sat_score = np.random.normal(1200, 150, n_samples).astype(int)
    sat_score = np.clip(sat_score, 800, 1600)

    # Extracurriculars and other factors
    extracurriculars = np.random.poisson(3, n_samples)
    extracurriculars = np.clip(extracurriculars, 0, 10)

    legacy = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

    family_income = np.random.choice(
        ["Low", "Middle", "High"], n_samples, p=[0.3, 0.5, 0.2]
    )

    # Generate admission with bias
    base_prob = 0.25

    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (gpa > 3.7) * 0.3
    prob_adjustments += (gpa > 3.5) * 0.15
    prob_adjustments += (sat_score > 1400) * 0.25
    prob_adjustments += (sat_score > 1300) * 0.15
    prob_adjustments += (extracurriculars > 4) * 0.1
    prob_adjustments += legacy * 0.2  # Legacy advantage
    prob_adjustments += (family_income == "High") * 0.08  # Wealth advantage
    prob_adjustments += (race == "Asian") * 0.05  # But also face discrimination
    prob_adjustments -= (race == "Asian") * 0.08  # Discrimination in admissions

    final_probs = np.clip(base_prob + prob_adjustments, 0.02, 0.95)
    admitted = np.random.binomial(1, final_probs)

    # Generate predictions
    admitted_pred = admitted.copy()
    error_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    admitted_pred[error_indices] = 1 - admitted_pred[error_indices]

    df = pd.DataFrame(
        {
            "Race": race,
            "Gender": gender,
            "GPA": gpa.round(2),
            "SATScore": sat_score,
            "Extracurriculars": extracurriculars,
            "Legacy": legacy,
            "FamilyIncome": family_income,
            "Admitted": admitted,
            "AdmittedPred": admitted_pred,
        }
    )

    return df


def generate_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Generate all test datasets.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing all generated datasets
    """
    datasets = {
        "heart_disease": create_heart_disease_dataset(),
        "hiring": create_hiring_dataset(),
        "loan_approval": create_loan_approval_dataset(),
        "criminal_justice": create_criminal_justice_dataset(),
        "college_admission": create_college_admission_dataset(),
    }

    return datasets


def save_datasets(
    datasets: dict[str, pd.DataFrame], output_dir: str = "tests/data"
) -> None:
    """
    Save all datasets to CSV files.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dictionary of datasets to save
    output_dir : str
        Directory to save the datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, df in datasets.items():
        file_path = output_path / f"{name}_data.csv"
        df.to_csv(file_path, index=False)
        print(
            f"Saved {name} dataset: {df.shape[0]} rows, {df.shape[1]} columns -> {file_path}"
        )

        # Print basic statistics
        print("  Sample data preview:")
        print(f"  {df.head(2).to_string()}")
        print()


def create_dataset_documentation() -> str:
    """
    Create documentation for the generated datasets.

    Returns
    -------
    str
        Documentation string
    """
    doc = """
# Generated Test Datasets for Fair-Mango Interface

This document describes the synthetic datasets generated for testing the Fair-Mango interface.
All datasets include realistic bias patterns and prediction errors to thoroughly test fairness metrics.

## Datasets Overview

### 1. Heart Disease Dataset (`heart_data.csv`)
**Purpose**: Medical diagnosis fairness testing
**Size**: ~918 samples
**Sensitive Attributes**: Sex, ChestPainType
**Target Variables**: HeartDisease, ExerciseAngina
**Predictions**: HeartDiseasePred, ExerciseAnginaPred
**Bias Patterns**: Gender bias in diagnosis

**Columns**:
- Age: Patient age (28-77)
- Sex: M/F
- ChestPainType: ASY/NAP/ATA/TA
- RestingBP: Resting blood pressure
- Cholesterol: Cholesterol level
- FastingBS: Fasting blood sugar (0/1)
- RestingECG: Normal/ST/LVH
- MaxHR: Maximum heart rate
- ExerciseAngina: Y/N
- Oldpeak: ST depression
- ST_Slope: Up/Flat/Down
- HeartDisease: Target (0/1)
- HeartDiseasePred: Prediction (0/1)
- ExerciseAnginaPred: Exercise angina prediction

### 2. Hiring Dataset (`hiring_data.csv`)
**Purpose**: Employment fairness testing
**Size**: 1000 samples
**Sensitive Attributes**: Gender, Race, AgeGroup
**Target Variable**: Hired
**Predictions**: HiredPred
**Bias Patterns**: Gender and racial bias in hiring

**Columns**:
- Gender: Male/Female
- Race: White/Black/Hispanic/Asian/Other
- AgeGroup: 18-30/31-45/46-60/60+
- Education: High School/Bachelor/Master/PhD
- Experience: Years of experience
- SkillsScore: Skills assessment (0-100)
- InterviewScore: Interview performance (0-100)
- Hired: Target (0/1)
- HiredPred: Prediction (0/1)

### 3. Loan Approval Dataset (`loan_approval_data.csv`)
**Purpose**: Financial fairness testing
**Size**: 1500 samples
**Sensitive Attributes**: Gender, Race
**Target Variable**: Approved
**Predictions**: ApprovedPred
**Bias Patterns**: Subtle demographic bias in lending

**Columns**:
- Gender: M/F
- Race: White/Black/Hispanic/Asian
- Age: Applicant age
- Income: Annual income
- CreditScore: Credit score (300-850)
- LoanAmount: Requested loan amount
- DebtToIncome: Debt-to-income ratio
- EmploymentLength: Years employed
- Approved: Target (0/1)
- ApprovedPred: Prediction (0/1)

### 4. Criminal Justice Dataset (`criminal_justice_data.csv`)
**Purpose**: Criminal justice fairness testing
**Size**: 800 samples
**Sensitive Attributes**: Race, Gender
**Target Variable**: Recidivism
**Predictions**: RecidivismPred
**Bias Patterns**: Systemic bias in recidivism prediction

**Columns**:
- Race: White/Black/Hispanic/Other
- Gender: Male/Female
- Age: Defendant age
- PriorConvictions: Number of prior convictions
- CrimeType: Violent/Property/Drug/Other
- SentenceLength: Sentence length in years
- EducationLevel: Education level
- EmploymentStatus: Employment status
- Recidivism: Target (0/1)
- RecidivismPred: Prediction (0/1)

### 5. College Admission Dataset (`college_admission_data.csv`)
**Purpose**: Educational fairness testing
**Size**: 1200 samples
**Sensitive Attributes**: Race, Gender
**Target Variable**: Admitted
**Predictions**: AdmittedPred
**Bias Patterns**: Admission bias and legacy advantages

**Columns**:
- Race: White/Asian/Hispanic/Black/Other
- Gender: Male/Female
- GPA: Grade point average (2.0-4.0)
- SATScore: SAT score (800-1600)
- Extracurriculars: Number of extracurricular activities
- Legacy: Legacy status (0/1)
- FamilyIncome: Low/Middle/High
- Admitted: Target (0/1)
- AdmittedPred: Prediction (0/1)

## Usage Examples

### Basic Dataset Loading
```python
import pandas as pd
from fair_mango.dataset.dataset import Dataset

# Load heart disease data
df = pd.read_csv("tests/data/heart_data.csv")
dataset = Dataset(df, ["Sex"], "HeartDisease", "HeartDiseasePred")

# Load hiring data with multiple sensitive attributes
df_hiring = pd.read_csv("tests/data/hiring_data.csv")
dataset_hiring = Dataset(df_hiring, ["Gender", "Race"], "Hired", "HiredPred")
```

### Testing Different Scenarios
```python
# Single sensitive attribute
dataset1 = Dataset(df, "Sex", "HeartDisease")

# Multiple sensitive attributes
dataset2 = Dataset(df, ["Sex", "ChestPainType"], "HeartDisease", "HeartDiseasePred")

# With positive target specified
dataset3 = Dataset(df, "Sex", "HeartDisease", "HeartDiseasePred", 1)

# Different target variable
dataset4 = Dataset(df, "Sex", "ExerciseAngina", "ExerciseAnginaPred", "Y")
```

## Data Quality Notes

- All datasets include realistic correlations between features
- Prediction accuracy varies by dataset (85-90% typical)
- Bias patterns are subtle but detectable by fairness metrics
- Sample sizes are appropriate for statistical significance
- Missing values are avoided for simplicity
- All categorical variables use consistent encoding

## Regeneration

To regenerate the datasets with different random seeds:
```bash
python generate_fake_data.py
```

The script uses fixed random seeds for reproducibility but can be modified for different data patterns.
"""
    return doc


if __name__ == "__main__":
    print("Generating fake datasets for Fair-Mango interface testing...")
    print("=" * 60)

    # Generate all datasets
    datasets = generate_all_datasets()

    # Save datasets
    save_datasets(datasets)

    # Create and save documentation
    doc = create_dataset_documentation()
    with open("tests/data/README.md", "w") as f:
        f.write(doc)

    print("Documentation saved to tests/data/README.md")
    print("\nDataset generation complete!")
    print("\nGenerated datasets:")
    for name, df in datasets.items():
        print(f"  - {name}_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nAll datasets are ready for interface testing!")
