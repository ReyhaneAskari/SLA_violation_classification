# SLA_violation_classification

This is the code for the paper: SLA Violation Prediction In Cloud Computing: A Machine Learning Perspective: https://arxiv.org/pdf/1611.10338.pdf

Cloud computing reduces the maintenance costs of services and allows users to access on demand services without being involved in technical implementation details. The relationship between a cloud provider and a customer is governed with a \textit{Service Level Agreement} (SLA) that is established to define the level of the service and its associated costs. SLA usually contains specific parameters and a minimum level of quality for each element of the service that is negotiated between a cloud provider and a customer.  The failure of providing the service is called an \textit{SLA violation}.

From a provider's point of view, since penalties have to be paid in case of SLA violation, violations prediction is an essential task. By predicting violations, the provider can reallocate the requests and prevent the violation. On the other hand, and from customer's point of view, predicting the future violations can be equivalent to provider's is trustworthiness. Also, the customer would like to receive the service on demand and without any interruptions. Despite the high availability rates, violations do happen in real world and have caused both the provider and the customer heavy costs. Thus, being able to predict SLA violations favors both the customers and the providers. 


To tackle this problem, one can use machine learning models to predict violations. Violation prediction task can be seen as a classification problem. Using a classifier, we can predict whether a coming request will be violated or not. In this work, we explore two machine learning models: Naive Bayes and Random Forest Classifiers to predict SLA violations. Unlike previous works on SLA violation prediction or avoidance, our models are trained on a real world dataset which introduces new challenges that have been neglected in previous works. We test our models using \textit{Google Cloud Cluster trace} as the dataset. This dataset contains 29-day trace of Google's Cloud Compute and was published on 2011.

Since SLA violations are rare events in real world ($\sim 0.2\%$), the classification task becomes more challenging because the classifier will always have the tendency to predict the dominant class. In order to overcome this issue, we use several re-sampling methods such as Random Over and Under Sampling, SMOTH, NearMiss (1,2,3), One-sided Selection, Neighborhood Cleaning Rule, etc. to re-balance the dataset. 

We demonstrate that Random forest with SMOTE-ENN re-sampling technique achieves the best performance among other methods with the accuracy of 0.9988% and $F_1$ score of 0.9980. Ensemble methods such as SMOTE-ENN overcome the problem of overfitting by re-sampling over classes. Random Forest with its tree based structure is less sensitive to class distributions. Thus, even with no re-sampling technique, it has an acceptable performance (accuracy = 0.97\% and $f_1$ = 0.79). On the other hand, Naive Bayes classifiers are highly biased with class distribution and do not have acceptable results without re-sampling techniques. It is worth mentioning that the random forest model has human-interpretable results which suggests the most important feature causing the violations.

## Citaion
If you use this code, please cite us here:


`
@article{hemmat2016sla,
  title={SLA violation prediction in cloud computing: A machine learning perspective},
  
  author={Hemmat, Reyhane Askari and Hafid, Abdelhakim},
  
  journal={arXiv preprint arXiv:1611.10338},
  
  year={2016}
}
`
