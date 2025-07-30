
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
