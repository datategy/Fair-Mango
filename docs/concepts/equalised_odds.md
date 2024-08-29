# What is Equalised Odds?

Equalised Odds is used to assess whether a machine learning model provides equal predictive performance across different demographic groups. It extends the concept of Equal Opportunity by considering both the True Positive Rate (TPR) and the False Positive Rate (FPR), ensuring fairness in both types of metrics across groups.

---

## How to calculate it?

Equalized Odds is calculated and applied following the next steps:

- **Define the Sensitive Attribute:** Identify the demographic attribute (e.g., race, gender) that is considered sensitive and protected from discrimination.
- **Group Data by Attribute:** Split the dataset into groups based on the sensitive attribute, creating distinct demographic groups (e.g male data/female data).
- **Train the Model:** Develop a machine learning model using the training data.
- **Predict Outcomes:** Apply the trained model to make predictions on the test data or real-world scenarios.
- **Calculate True Positive Rate (TPR) and False Positive Rate (FPR):** For each de- mographic group, calculate both the TPR (proportion of actual positive cases correctly predicted as positive) and the FPR (proportion of actual positive cases incorrectly predicted as negative) based on the model’s predictions.
- **Compare TPR and FPR Across Groups:** Examine both the TPR and FPR for each demographic group and compare them.
- **Assess Parity:** Equalized Odds is achieved when both the TPR and FPR are similar across all demographic groups.
---

## How to interpret its value?

### Using Difference

- The values will be in the range of **[-1, 1]**.
- A value of **0** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* from **0** will define the maximum disparity in the True Positive Rate or False Positive Rate between the two compared groups. The sign will indicate which group is privileged and which group is being discriminated.

### Using Ratio

- The values will be in the range of **]-∞, ∞[**.
- A value of **1** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* then **1** will define the maximum disparity in the True Positive Rate or False Positive Rate between the two compared groups. The value compared to **-1** and **1** will indicate which group is privileged and which group is being discriminated.

---

## Example using FairMango

Checkout the API documentation for [equalised odds difference](../api_documentation/metrics.md#fair_mango.metrics.metrics.EqualisedOddsDifference) and for [equalised odds ration](../api_documentation/metrics.md#fair_mango.metrics.metrics.EqualisedOddsRatio).
