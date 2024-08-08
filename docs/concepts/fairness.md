# What is Fairness in Machine Learning and AI?

Artificial Intelligence (AI) systems have become integral to many aspects of modern life, influencing decisions in areas such as healthcare, finance, hiring, and law enforcement. As AI systems grow in capability and deployment, ensuring their fairness becomes critically important. Fairness in AI aims to prevent and mitigate biases that could lead to unfair or discriminatory outcomes. This page provides an overview of key concepts and definitions related to fairness in AI.

---

## Fairness

Fairness in AI refers to the principle that AI systems should make decisions that are just, equitable, and free from bias. This includes ensuring that these systems do not perpetuate existing social inequalities or introduce new forms of discrimination. Fairness can be considered from multiple perspectives, including distributive fairness (how resources and opportunities are distributed), procedural fairness (the processes by which decisions are made), and representational fairness (how individuals and groups are represented).

---

## Key Objectives of Fairness in AI

1. **Preventing Discrimination:** Ensuring that AI systems do not make decisions based on inappropriate factors such as race, gender, age, or other protected characteristics.
2. **Promoting Equity:** Providing equal opportunities for all individuals, particularly those from historically marginalized or underrepresented groups.
3. **Transparency and Accountability:** Ensuring that AI systems are transparent in their decision-making processes and that there are mechanisms for accountability when unfair outcomes occur.

---

## Sensitive Variables

Sensitive variables, also known as protected variables, are characteristics of individuals that are legally or ethically recognized as grounds for protection against discrimination. These variables often include, but are not limited to, race, gender, age, religion and disability status. In the context of AI, it is crucial to handle these variables carefully to prevent biased outcomes.

Examples of Sensitive Variables:

- **Race and Ethnicity:** Categories that reflect an individual's heritage and cultural background.
- **Gender:** Identifying characteristics related to an individual's sex or gender identity.
- **Age:** An individual's age, often protected to prevent age discrimination.
- **Religion:** An individual's religious beliefs or affiliations.
- **Disability Status:** Whether an individual has a physical or mental impairment that substantially limits one or more major life activities.

---

## Proxy Variables

Proxy variables are variables that, while not directly representing a sensitive attribute, correlate strongly with it and can indirectly introduce bias into AI systems. For instance, a variable such as ZIP code might serve as a proxy for race or socioeconomic status due to the demographic composition of certain areas. The use of proxy variables can lead to unfair outcomes, as decisions may inadvertently reflect the biases associated with the sensitive attribute they correlate with.

### Examples of Proxy Variables

| Sensitive Variable      | Potential Proxies                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Gender                  | Education Level, Income, Occupation, Regular purchases, University Faculty, Working Hours |
| Race                    | Felony Data, Zipcode                                                                      |
| Age                     | Education Level, Years of Experience, Occupation                                          |
| Disabilities            | Health Insurance Status, Number of Medical Visits                                         |

### Mitigating the Impact of Proxy Variables

- **Detection and Analysis:** Regularly analyzing and detecting proxy variables within the data to understand their impact on model outcomes.
- **Fairness Constraints:** Implementing fairness constraints in the algorithm to mitigate the influence of proxy variables.
- **Data Anonymization:** Anonymizing or aggregating data to reduce the risk of proxy variables influencing decisions.
- **Alternative Variables:** Identifying and using alternative variables that do not correlate with sensitive attributes while still providing necessary information for decision-making.

!!! note

    Handling proxy variables is not supported on FairMango yet. Maybe you can help us implementing it! Check out our [contribution guide](../contribution_guide.md).

---

## Algorithmic Bias

Algorithmic bias occurs when an AI system systematically and unfairly discriminates against certain individuals or groups based on sensitive variables. This bias can arise from various sources, including biased training data, flawed algorithms, or the misapplication of algorithms. Understanding and mitigating algorithmic bias is essential for ensuring fairness in AI systems.

---

## Types of Algorithmic Bias

- **Data reflects hidden biases in society:** for example if an AI is trained using recent news articles or books, the word ’nurse’ is more likely to refer to a woman and the word ’programmer’ is more likely to refer to a man. The same thing is observed when searching on Google Images. This shows how hidden biases in the data get embedded in AI models, which in turn may spread those biases to more human brains. Using training data that does not group individuals based on protected variables like gender or race, will not ensure a fair model because these protected variables may emerge as correlated features through proxy variables.
- **Unbalanced training data:** the training data may not have enough examples of each class, which leads to a model that performs well on a the majority class and performs badly on the minority class. A solid example is facial recognition AI algorithms that are trained using data that includes a higher proportion of white peoples’ faces compared to other races. According to Reuters, a passport photo checker AI used by New Zealand’s department of internal affairs in 2016 mistakenly registered the eyes of Asian descents as closed. Another example is a candidate evaluator system that unfairly recommends male candidates over equally-qualified female candidates for executive positions simply because those positions are dominated by men.
- **Data is not quantifiable:** Numbers provide a standardized, objective way to compare and analyze data, enabling the identification of patterns and trends. Sometimes, it is difficult to quantify or measure features in the data; Like relationship strength, trust, quality of service. Basically, any element that is related to sentiment or personal preference is not represented as numbers easily (that's if it is possible in the first place). Efforts to put these subtle qualities into numbers have challenges because they are influenced by subjective experiences and perspectives. Hence, evaluating them properly needs a mix of understanding their context and using both numbers and qualitative insights to grasp their full meaning and importance.
One recent example is trying to use AI to grade writing on standardized tests like SATs and GREs with the goal to save human grader’s time. However, good writing involves intricate elements like clarity, structure and creativity but most of these qualities are hard to measure. Consequently, the AI was trained using easier-to-measure elements like sentence length, vocabulary and grammar which do not fully represent good writing and made the AI susceptible to manipulation. An instance of this is BABEL Generator, a natural language program that generates nonsensical essays nonetheless receive high ratings by these AI grading algorithms.
- **Data amplified by a feedback loop:** The algorithm can influence the data that it gets, establishing a feedback loop that tends to reinforce historical patterns, irrespective of their desirability. An example is PredPol’s crime prediction algorithm, which has been operational since 2012 across several major cities in the United States, including Los Angeles and Chicago. The model was trained on data that was heavily biased by past housing segregation and cases of police bias. The model exhibits a tendency to disproportionately target neighborhoods with high concentrations of racial minority populations. Subsequently, these new arrest statistics are incorporated by the algorithm, leading to a further prediction of increased drug-related arrests in the same neighborhoods and perpetuating the cycle. Even though there might be criminal activity in places where police were not being sent by this AI, simply due to the absence of prior arrests
