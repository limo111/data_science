

# # Regularization with SciKit-Learn
# 
# Previously we created a new polynomial feature set and then applied our standard linear regression on it, but we can be smarter about model choice and utilize regularization.
# 
# Regularization attempts to minimize the RSS (residual sum of squares) *and* a penalty factor. This penalty factor will penalize models that have coefficients that are too large. Some methods of regularization will actually cause non useful features to have a coefficient of zero, in which case the model does not consider the feature.
# 
# Let's explore two methods of regularization, Ridge Regression and Lasso. We'll combine these with the polynomial feature set (it wouldn't be as effective to perform regularization of a model on such a small original feature set of the original X).

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Data and Setup

# In[6]:


df = pd.read_csv("Advertising.csv")
X = df.drop('sales',axis=1)
y = df['sales']


# ### Polynomial Conversion

# In[7]:


from sklearn.preprocessing import PolynomialFeatures


# In[8]:


polynomial_converter = PolynomialFeatures(degree=3,include_bias=False)


# In[9]:


poly_features = polynomial_converter.fit_transform(X)


# ### Train | Test Split

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)


# ----
# ----
# 
# ## Scaling the Data
# 
# While our particular data set has all the values in the same order of magnitude ($1000s of dollars spent), typically that won't be the case on a dataset, and since the mathematics behind regularized models will sum coefficients together, its important to standardize the features. Review the theory videos for more info, as well as a discussion on why we only **fit** to the training data, and **transform** on both sets separately.

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


# help(StandardScaler)


# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X_train)


# In[16]:


X_train = scaler.transform(X_train)


# In[17]:


X_test = scaler.transform(X_test)


# ## Ridge Regression
# 
### Ridge regression is a regularization technique that adds a penalty term to the cost function in linear regression. The penalty term is based on the L2 norm of the coefficients, which is the sum of the squared values of the coefficients.
## J(w) = RSS(w) + alpha * ||w||^2
### where J(w) is the cost function, RSS(w) is the residual sum of squares (i.e., the sum of the squared differences between the predicted values and the actual values), w is the vector of coefficients, ||w||^2 is the L2 norm of the coefficients, and alpha is a hyperparameter that controls the strength of the regularization.

### The L2 penalty term alpha * ||w||^2 is added to the cost function to shrink the coefficients towards zero, which helps to reduce overfitting and improve the generalization performance of the model. The larger the value of alpha, the stronger the penalty, and the more the coefficients are shrunk towards zero.

### During the training process, the ridge regression algorithm tries to find the values of the coefficients that minimize the cost function. The addition of the penalty term in the cost function encourages the algorithm to find a solution with smaller coefficients, which can help to reduce the impact of noisy or irrelevant features in the dataset.

# In[18]:


from sklearn.linear_model import Ridge


# In[19]:


ridge_model = Ridge(alpha=10)


# In[20]:


ridge_model.fit(X_train,y_train)


# In[21]:


test_predictions = ridge_model.predict(X_test)


# In[22]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[23]:


MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)


# In[24]:


MAE


# In[25]:


RMSE


# How did it perform on the training set? (This will be used later on for comparison)

# In[26]:


# Training Set Performance
train_predictions = ridge_model.predict(X_train)
MAE = mean_absolute_error(y_train,train_predictions)
MAE


# ### Choosing an alpha value with Cross-Validation
# 
# Review the theory video for full details.

# In[27]:


from sklearn.linear_model import RidgeCV


# In[28]:


# help(RidgeCV)


# In[29]:


### Ridge regression with cross-validation (CV) is a common technique used to find the optimal value of the regularization parameter, alpha, in ridge regression.

### The basic idea behind this technique is to split the data into k folds and use k-1 folds for training and the remaining fold for validation. This process is repeated k times, with each fold used once for validation.

### For each value of alpha that we want to test, we perform k-fold cross-validation and compute the average validation error. We then choose the value of alpha that minimizes the average validation error. This approach helps to avoid overfitting and provides a more reliable estimate of the performance of the model on new, unseen data.



 
# Negative RMSE so all metrics follow convention "Higher is better"

# See all options: sklearn.metrics.SCORERS.keys()
ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')


# In[30]:


# The more alpha options you pass, the longer this will take.
# Fortunately our data set is still pretty small
ridge_cv_model.fit(X_train,y_train)


# In[31]:


ridge_cv_model.alpha_


# In[32]:


test_predictions = ridge_cv_model.predict(X_test)


# In[33]:


MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)


# In[34]:


MAE


# In[35]:


RMSE


# In[36]:


# Training Set Performance
# Training Set Performance
train_predictions = ridge_cv_model.predict(X_train)
MAE = mean_absolute_error(y_train,train_predictions)
MAE


# In[37]:


ridge_cv_model.coef_


# 
# -----
# 
# ## Lasso Regression

### Lasso (Least Absolute Shrinkage and Selection Operator) is a regression technique that adds a penalty term to the cost function in linear regression. The penalty term is based on the L1 norm of the coefficients, which is the sum of the absolute values of the coefficients.
## J(w) = RSS(w) + alpha * ||w||_1
### where J(w) is the cost function, RSS(w) is the residual sum of squares (i.e., the sum of the squared differences between the predicted values and the actual values), w is the vector of coefficients, ||w||_1 is the L1 norm of the coefficients, and alpha is a hyperparameter that controls the strength of the regularization.

### The L1 penalty term alpha * ||w||_1 is added to the cost function to shrink the coefficients towards zero, which helps to reduce overfitting and improve the generalization performance of the model. Like ridge regression, lasso regression also encourages sparse solutions, where some of the coefficients are exactly zero.

### During the training process, the lasso regression algorithm tries to find the values of the coefficients that minimize the cost function. The addition of the penalty term in the cost function encourages the algorithm to find a solution with smaller coefficients, which can help to reduce the impact of noisy or irrelevant features in the dataset.

### Lasso regression is particularly useful when we have a large number of features and we want to perform feature selection by automatically setting some of the coefficients to zero. However, lasso regression can be more sensitive to the scale of the features compared to ridge regression. To mitigate this issue, we can standardize the features before applying lasso regression.

from sklearn.linear_model import LassoCV


# In[39]:



lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)


# In[40]:


lasso_cv_model.fit(X_train,y_train)


# In[41]:


lasso_cv_model.alpha_


# In[42]:


test_predictions = lasso_cv_model.predict(X_test)


# In[43]:


MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)


# In[44]:


MAE


# In[45]:


RMSE


# In[46]:


# Training Set Performance
# Training Set Performance
train_predictions = lasso_cv_model.predict(X_train)
MAE = mean_absolute_error(y_train,train_predictions)
MAE


# In[47]:


lasso_cv_model.coef_


# ## Elastic Net
# 
# Elastic Net combines the penalties of ridge regression and lasso in an attempt to get the best of both worlds!
### Elastic Net is a regularization technique that combines the L1 and L2 penalties of Lasso and Ridge regression, respectively. Elastic Net adds a penalty term to the cost function in linear regression that is a linear combination of both L1 and L2 penalties
## J(w) = RSS(w) + alpha * rho * ||w||_1 + 0.5 * alpha * (1 - rho) * ||w||_2^2
### where J(w) is the cost function, RSS(w) is the residual sum of squares, w is the vector of coefficients, ||w||_1 is the L1 norm of the coefficients, ||w||_2 is the L2 norm of the coefficients, alpha is a hyperparameter that controls the strength of the regularization, and rho is a hyperparameter that controls the balance between L1 and L2 penalties.

### The L1 penalty encourages sparsity in the coefficients, while the L2 penalty encourages small but non-zero coefficients. The choice of rho determines the balance between the two penalties, and it can be tuned using cross-validation.

### Elastic Net can be a good compromise between Ridge and Lasso regression, as it can handle correlated features better than Lasso regression and can select relevant features better than Ridge regression. It is especially useful when the number of features is larger than the number of observations or when the features are highly correlated.
# In[48]:


from sklearn.linear_model import ElasticNetCV


# In[51]:


elastic_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)


# In[52]:


elastic_model.fit(X_train,y_train)


# In[55]:


elastic_model.l1_ratio_


# In[56]:


test_predictions = elastic_model.predict(X_test)


# In[57]:


MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)


# In[58]:


MAE


# In[59]:


RMSE


# In[60]:


# Training Set Performance
# Training Set Performance
train_predictions = elastic_model.predict(X_train)
MAE = mean_absolute_error(y_train,train_predictions)
MAE


# In[61]:


elastic_model.coef_


# -----
# ---
