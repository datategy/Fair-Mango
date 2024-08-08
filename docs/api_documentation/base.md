
This documentation focuses on the foundational base classes that underpin the fairness metrics, detailing the core calculations. These base classes are designed to handle complex computations and data manipulations, ensuring that the derived fairness metrics classes can seamlessly provide the results. There are three main base classes:

1. **FairnessMetricDifference:** This class is inherited by fairness metrics that are based of the `difference` operation to calculate the disparity between the different sensitive groups.
2. **FairnessMetricRatio:** This class is inherited by fairness metrics that are based of the `ratio` operation to calculate the disparity between the different sensitive groups.
3. **Metric:** This class is inherited by performance evaluation classes giving a more detailed results by assigning a score to every sensitive group.

---

::: fair_mango.metrics.base
