
### Basic Machine Learning Concepts

**Q1:** What is Machine Learning (ML)?
**A1:** ML is a field of artificial intelligence that allows systems to learn from data and improve their performance without being explicitly programmed.

**Q2:** What are outliers?
**A2:** Outliers are data points that differ significantly from other observations and may indicate variability in measurement, experimental errors, or novel phenomena.

**Q3:** What is a correlation matrix?
**A3:** A correlation matrix is a table showing correlation coefficients between many variables. It helps identify the strength and direction of relationships.

**Q4:** What is the difference between supervised and unsupervised learning?
**A4:** Supervised learning uses labeled data to train models, while unsupervised learning finds hidden patterns or intrinsic structures in unlabeled data.

**Q5:** What is overfitting in ML?
**A5:** Overfitting occurs when a model learns the training data too well, including its noise, leading to poor performance on new data.

**Q6:** What is underfitting?
**A6:** Underfitting happens when a model is too simple to capture the underlying patterns in data, resulting in poor performance on both training and test data.

**Q7:** What is a feature in ML?
**A7:** A feature is an individual measurable property or characteristic used as input in a model.

**Q8:** What is model accuracy?
**A8:** Accuracy is a metric that shows the percentage of correct predictions made by the model over all predictions.

**Q9:** What is a dataset?
**A9:** A dataset is a collection of data used to train or test a machine learning model.

**Q10:** What is dimensionality reduction?
**A10:** Dimensionality reduction is the process of reducing the number of features or variables under consideration to simplify models and avoid overfitting.

**Q11:** What is normalization and why is it important?
**A11:** Normalization scales numerical features to a standard range, improving convergence and performance of many ML algorithms.

**Q12:** What is cross-validation?
**A12:** Cross-validation is a technique to evaluate model performance by dividing the dataset into training and testing sets multiple times.

**Q13:** What is PCA (Principal Component Analysis)?
**A13:** PCA is a dimensionality reduction technique that transforms correlated features into a set of linearly uncorrelated components while preserving as much variance as possible.

**Q14:** What is the Find-S algorithm?
**A14:** The Find-S algorithm is a concept learning algorithm that finds the most specific hypothesis consistent with all positive examples in the training data.

**Q15:** What is a hypothesis in ML?
**A15:** A hypothesis is a function or model that makes predictions based on input data.

**Q16:** What is k-Nearest Neighbours (k-NN)?
**A16:** k-NN is a simple classification algorithm that assigns a class to a data point based on the majority class among its k nearest neighbors.

**Q17:** What is non-parametric Locally Weighted Regression (LWR)?
**A17:** LWR is a regression method that fits a model locally using a weighted subset of data around the target point, without assuming a global model form.

**Q18:** What is a prediction in ML?
**A18:** A prediction is the model’s output or inference based on input features.

**Q19:** What is regression?
**A19:** Regression is a type of supervised learning used to predict continuous output variables.

**Q20:** What is linear regression?
**A20:** Linear regression is a statistical method that models the linear relationship between a dependent variable and one or more independent variables.

**Q21:** What is polynomial regression?
**A21:** Polynomial regression is a type of regression where the relationship between the independent variable and dependent variable is modeled as an nth-degree polynomial.

**Q22:** What is classification?
**A22:** Classification is a supervised learning task where the output variable is a category or class label.

**Q23:** What is the decision tree algorithm?
**A23:** A decision tree is a flowchart-like structure used for classification and regression, which splits the data based on feature values to arrive at a decision.

**Q24:** What is Naive Bayes?
**A24:** Naive Bayes is a probabilistic classification technique based on Bayes’ theorem, assuming feature independence.

**Q25:** What is a classifier?
**A25:** A classifier is an algorithm that maps input data to a category label.

**Q26:** What is k-means?
**A26:** k-means is an unsupervised clustering algorithm that partitions data into k clusters by minimizing variance within each cluster.

**Q27:** What is clustering?
**A27:** Clustering is the task of grouping a set of objects so that objects in the same group are more similar to each other than to those in other groups.

---

**LAB PROGRAM QUESTIONS AND SOLUTIONS**

---

### 1. Data Visualization using California Housing Dataset

**Q1:** What are histograms used for in data analysis?
**A1:** Histograms visualize the frequency distribution of numerical data, helping to understand the spread and central tendency.

**Q2:** How do you identify outliers using a box plot?
**A2:** Outliers are shown as points outside the whiskers of the box plot, typically beyond 1.5 IQRs from the quartiles.

**Q3:** What libraries are used in this program?
**A3:** `matplotlib`, `pandas`, `seaborn`, and `sklearn.datasets`.

---

### 2. Correlation Analysis

**Q1:** How do you compute the correlation matrix?
**A1:** Using `df.corr()` in pandas.

**Q2:** What does a heatmap of a correlation matrix show?
**A2:** It shows the strength and direction of correlations between features using color intensities.

**Q3:** How is a pair plot helpful?
**A3:** It provides scatter plots of feature pairs, helping to visualize relationships and class separations.

---

### 3. PCA on Iris Dataset

**Q1:** What is PCA used for?
**A1:** PCA reduces the dimensionality of data while preserving variance.

**Q2:** Why reduce from 4 to 2 components?
**A2:** To visualize the data in 2D while retaining most variance.

**Q3:** What does `explained_variance_ratio_` indicate?
**A3:** The proportion of variance explained by each principal component.

---

### 4. Find-S Algorithm

**Q1:** What is the goal of Find-S algorithm?
**A1:** To find the most specific hypothesis consistent with all positive examples.

**Q2:** How are negative examples handled?
**A2:** They are ignored in Find-S algorithm.

**Q3:** What does the final hypothesis look like?
**A3:** A list of specific values and '?' where generalization occurs.

---

### 5. k-Nearest Neighbour Classification

**Q1:** How does k-NN classify a new point?
**A1:** By majority vote among the k nearest neighbors.

**Q2:** What happens when you increase k?
**A2:** Classification becomes smoother but may miss finer distinctions.

**Q3:** What evaluation metric is used?
**A3:** Accuracy score using `knn.score()`.

---

### 6. Locally Weighted Regression (LWR)

**Q1:** What is LWR?
**A1:** A non-parametric regression that fits a model locally around a target point.

**Q2:** How are weights determined?
**A2:** Using a Gaussian kernel based on distance to the test point.

**Q3:** What is the role of tau?
**A3:** It controls the bandwidth or locality of the regression.

---

### 7. Linear vs Polynomial Regression

**Q1:** What dataset is used for Linear Regression?
**A1:** Boston Housing Dataset.

**Q2:** What is the regression formula used?
**A2:** `y = b0 + b1*x`, where x is average number of rooms.

**Q3:** What dataset is used for Polynomial Regression?
**A3:** Auto MPG dataset.

**Q4:** Why use polynomial regression here?
**A4:** To model nonlinear relationships between horsepower and MPG.

---

### 8. Decision Tree Classification

**Q1:** What dataset is used for classification?
**A1:** Breast Cancer Wisconsin dataset.

**Q2:** What criteria is used for splitting?
**A2:** Gini index.

**Q3:** How is the decision tree visualized?
**A3:** Using `plot_tree()` from `sklearn.tree`.

**Q4:** How is a new sample classified?
**A4:** By passing it through the trained classifier using `predict()`.

---

### 9. Naive Bayes Classifier

**Q1:** What dataset is used?
**A1:** Olivetti Face Dataset.

**Q2:** What Naive Bayes model is used?
**A2:** GaussianNB.

**Q3:** How is accuracy evaluated?
**A3:** Using `accuracy_score()` comparing predicted and actual labels.

---

### 10. K-Means Clustering

**Q1:** What dataset is used for clustering?
**A1:** Breast Cancer dataset (WBCD-lab8.csv).

**Q2:** How are features scaled?
**A2:** Using `StandardScaler()`.

**Q3:** Why is PCA applied before visualization?
**A3:** To reduce dimensions for 2D plotting.

**Q4:** How is clustering visualized?
**A4:** Using `seaborn.scatterplot()` on PCA components.
