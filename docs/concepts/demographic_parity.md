# What is Demographic Parity?

Also known as Statistical Parity is a fairness metric used to assess whether the distribution of the actual dataset treats different sensitive groups equally in terms of outcomes. It focuses on ensuring that the proportion of positive outcomes is similar across different sensitive groups, regardless of the sensitive attributes such as race or gender.

---

## How to calculate it?

Demographic Parity is typically defined and calculated as:

- **Define the Sensitive Attribute:** Identify the demographic attribute (e.g., race, gender) that is considered sensitive and protected from discrimination.
- **Group Data by Attribute:** Split the dataset into groups based on the sensitive attribute, creating distinct sensitive groups.
- **Calculate Selection Rates(Positive outcome rates):** For each sensitive group, calculate the proportion of individuals who received positive outcomes in the dataset. This could be, for example, the proportion of approved loan applications, job offers, or accepted university admissions.
- **Compare Positive Outcome Rates Across Groups:** Examine the positive outcome rates for each sensitive group and compare them relative to each other and to the overall population positive outcome rate.
- **Assess Parity:** Demographic parity is achieved when the positive outcome rates are similar across all demographic groups.

---

## How to interpret its value?

### Using Difference

- The values will be in the range of **[-1, 1]**.
- A value of **0** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* from **0** will define the disparity in the selection rate **(using actual labels)** between the two compared groups. The sign will indicate which group is privileged and which group is being discriminated.

### Using Ratio

- The values will be in the range of **]-∞, ∞[**.
- A value of **1** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* then **1** will define the disparity in the selection rate **(using actual labels)** between the two compared groups. The value compared to **-1** and **1** will indicate which group is privileged and which group is being discriminated.

!!! note

    Demographic Parity uses the **actual labels** to calculate the Selection Rate.

---

## Example using FairMango

Checkout the API documentation for [demographic parity difference](/api_documentation/metrics/#fair_mango.metrics.metrics.DemographicParityDifference) and for [demographic parity ration](/api_documentation/metrics/#fair_mango.metrics.metrics.DemographicParityRatio).
