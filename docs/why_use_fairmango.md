# Why use fair mango ?

Fair mango is a state-of-the-art library to computes fairness across all the sensitive groups in datasets. Not only does it give you key dataset metrics, it also underlines how one sensitive group might be priviledged compared another.

# How does it compare to fairlearn ?

Fairlearn and fair mango have the same metrics available. However, fair mango compares fairness metrics for each combination of groups. For instance, if you are studing sensitive features `gender` (with values `male` and `female`) and `race` (with values `caucasian`, `north african`, `east asian`), fair mango will give you results for `caucasian males` vs `caucasian females`, `caucasian females` vs `north african males`, `east asian males` vs `north african females`, and so on...
