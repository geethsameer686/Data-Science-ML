Personalized cancer diagnosis

Type of Machine Learning Problem:

* There are nine different classes a genetic mutation can be classified into => Multi class classification problem

Performance Metric:

	* Multi class log-loss
        * Confusion matrix

Objectives and Constraints:

Objective: Predict the probability of each data-point belonging to each of the nine classes.
Constraints:
 * Interpretability
 * Class probabilities are needed.
 * Penalize the errors in class probabilites => Metric is Log-loss.
 * No Latency constraints
