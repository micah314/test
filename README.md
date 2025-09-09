[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/T0OyncOs)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13912005&assignment_repo_type=AssignmentRepo)
## CS 360 Lab 4 - Ensemble Methods

Name: Micah Painter 

Number of Late Days using for this lab: 0

---

### Analysis

1. Based on AUC, which method would you choose for this application?
Which threshold would you choose? Does your answer depend on how much training
time you have available?
    Based on the AUC, I would choose ada_boost becasue it is consistantly closer to 1, meaning that there is more area under the curve and thus that it is closer to the perfect curve. This does not depend on how much training data I have.
    As for threshold, I would choose a low threshold so that more things were classified as posionous than edible, because I'd rather not eat an edible mushroom than eat a posionous mushroom. This does not change based on training time.

2. `T` can be thought of as a measure of model complexity. Do these methods seem
to suffer from overfitting as the model complexity increases? What is the
intuition behind your answer?
    These models do not seems to suffer from overfitting. We can see this because the AUC values increase as T increases, even very dramatically, and don't seem to have an optimal point. For example, the AUC for random forest was 0.89 for T=5 and 0.91 for T=100. Furthermore, the AUC for adaBoost was 0.98 for T=5 and 0.99 for T=100. This means that there is not an optimal point at which the effectiveness of the model for the training data decreases, suggesting there is no overfitting.

---

### Lab Questionnaire

(None of your answers below will affect your grade; this is to help refine lab assignments in the future)

1. Approximately, how many hours did you take to complete this lab? (provide your answer as a single integer on the line below)
9

2. How difficult did you find this lab? (1-5, with 5 being very difficult and 1 being very easy)
4

3. Describe the biggest challenge you faced on this lab:
Figuring out how to use sklearn to get the ROC curve to work / figuring out what the y_score parameter was
