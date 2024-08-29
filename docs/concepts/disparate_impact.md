# What is Disparate Impact?

Disparate impact is a fairness metric used to assess whether a particular decision-making process or model has a disproportionate impact on different groups, especially those protected by anti-discrimination laws. It is calculated by comparing the outcomes of a decision across different sensitive groups and examining whether there are significant differences in these predictions.

---

## How to calculate it?

Disparate impact is typically calculated following these steps:

- **Define the Sensitive Attribute:** Identify a characteristic, such as race, gender, age, or ethnicity, that is protected from discrimination.
- **Group Data by Attribute:** Divide the dataset into groups based on the sensitive attribute. For example, job applicants might be divided into male and female groups.
- **Determine the Outcome:** Identify the outcome of interest, such as being hired for a job, receiving a loan, or being admitted to a program.
- **Calculate Outcome Rates:** For each group, calculate the proportion of individuals who experienced the desired outcome. This is done by dividing the number of individuals who experienced the outcome by the total number of individuals in that demographic group. It is often expressed as a percentage.
- **Compare Outcome Rates:** Compare the outcome rates across different demographic groups. If there is a substantial difference in outcome rates between groups, it suggests disparate impact. A common way to decide if there is Disparate Impact is the 80% rule that suggests if the ratio of outcome rates for the minority group is less than 80% of the outcome rates for the majority group; Then we say that there is Disparate Impact.

---

## How to interpret its value?

### Using Difference

- The values will be in the range of **[-1, 1]**.
- A value of **0** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* from **0** will define the disparity in the selection rate **(using predictions)** between the two compared groups. The sign will indicate which group is privileged and which group is being discriminated.

### Using Ratio

- The values will be in the range of **]-∞, ∞[**.
- A value of **1** means there is no disparity -> Fair outcomes -> No bias.
- A value *different* then **1** will define the disparity in the selection rate **(using predictions)** between the two compared groups. The value compared to **-1** and **1** will indicate which group is privileged and which group is being discriminated.

!!! note

    Disparate Impact uses the **predictions** to calculate the Selection Rate.

---

## Example using FairMango

Checkout the API documentation for [disparate impact difference](../api_documentation/metrics.md#fair_mango.metrics.metrics.DisparateImpactDifference) and for [disparate impact ration](../api_documentation/metrics.md#fair_mango.metrics.metrics.DisparateImpactRatio).
