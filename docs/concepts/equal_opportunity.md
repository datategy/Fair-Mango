# What is Equal Opportunity?

Equal Opportunity is a fairness metric used to evaluate the performance of a machine learning model with respect to a sensitive attribute, such as race or gender, specifically in binary classification tasks. It measures the difference in true positive rates (also known as sensitivity or recall) between the different sensitive groups.

---

## How to calculate it?

Equal Opportunity is calculated using a similar algorithm to disparate impact with the difference of using True Positive Rate instead of Selection Rate:

- **Define the Sensitive Attribute:** Identify the attribute that is considered sensi- tive, such as race or gender.
- **Group Data by Attribute:** Divide the dataset into groups based on the sensitive attribute, creating privileged and unprivileged groups.
- **Train the Model:** Develop a binary classification model using the training data. Predict Outcomes: Use the trained model to make predictions on the test data.
- **Calculate True Positive Rates (TPR):** For each group (privileged and unprivi- leged), calculate the true positive rate, which is the proportion of actual posi- tive cases correctly identified by the model within that group.
- **Calculate Equal Opportunity:** Calculate the disparity between the true positive rate of the different sensitive groups.

---

## How to interpret its value?

### Using Difference

- The values will be in the range of **[-1, 1]**.
- A value of **0** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* from **0** will define the disparity in the True Positive Rate between the two compared groups. The sign will indicate which group is privileged and which group is being discriminated.

### Using Ratio

- The values will be in the range of **]-∞, ∞[**.
- A value of **1** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* then **1** will define the disparity in the True Positive Rate between the two compared groups. The value compared to **-1** and **1** will indicate which group is privileged and which group is being discriminated.

---

## Example using FairMango

Checkout the API documentation for [equal opportunity difference](/api_documentation/metrics/#fair_mango.metrics.metrics.EqualOpportunityDifference) and for [equal opportunity ration](/api_documentation/metrics/#fair_mango.metrics.metrics.EqualOpportunityRatio).
