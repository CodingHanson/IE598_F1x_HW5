import pandas as pd
import numpy as np

df = pd.read_csv('F:/MSFE/machine_learning/HW5/hw5_treasury yield curve data.csv')
df = df.dropna()
df = df.drop(['Date'],axis=1)
y = df['Adj_Close'].values
X = df.drop(['Adj_Close'],axis=1)

#EDA
import matplotlib.pyplot as plt
import seaborn as sns
print('Number of rows of data:', df.shape[0])
print('Number of columns of data:', df.shape[1])
print(df.info())

percentiles = np.array([2.5,25,50,75,97.5])
ptiles_X = np.percentile(X,percentiles)
ptiles_y = np.percentile(y,percentiles)
print(ptiles_X )
summary_X = X.describe()
print("The summary of X:", summary_X)

print(ptiles_y )
summary_y = X.describe()
print("The summary of y:",summary_y)

sns.set()




y_n = len(y)
bins_2 = np.sqrt(y_n)
y_bins = int(bins_2)
y_ = plt.hist(y,bins=y_bins,density=True, facecolor = 'green', alpha=0.5)
plt.xlabel('target_value')
plt.ylabel('target_number')
plt.title('Histogram of target')
plt.show()
plt.clf()

# summary plot
cols = ['SVENF01','SVENF02','SVENF03','SVENF07','SVENF08','SVENF09','SVENF10','Adj_Close']
sns.pairplot(df[cols],size=4.5)
plt.tight_layout()
plt.show()
# heatmap

cm=np.corrcoef(df.values.T)
sns.set(font_scale=0.8)
sns.set_style("dark")
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size':8},yticklabels=df.columns,xticklabels=df.columns)
plt.show()
#box plot
cols = ['SVENF06','SVENF16']
fig,ax = plt.subplots(len(cols),figsize = (8,40))
for i, col_val in enumerate(cols):
    sns.boxplot(y=X[col_val],ax=ax[i])
    ax[i].set_title('Box Plot - {}'.format(col_val),fontsize=10)
    ax[i].set_xlabel(col_val,fontsize=8)
plt.show()

#PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn import preprocessing
import time
pca = PCA(n_components=3)
sc = StandardScaler()
lr = LinearRegression()
X_train, X_test,y_train,y_test = train_test_split( X , y ,test_size=0.15,random_state=42)
print('Shape of X_train: ', X_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_test: ', y_test.shape)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
#PCA
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


cov_mat = np.cov(X_test_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigenvalues:',eigen_vals)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
print('Explained variance ratio of the 3-component version: ', pca.explained_variance_ratio_)
print('Explained variance of the 3-component version: ', pca.explained_variance_)
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,31),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,31),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()


# Without PCA Logistic regression Classifier vs SVM Classifier
lr.fit(X_train,y_train)
start = time.clock()
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)
train_score_R2_lr = lr.score(X_train,y_train)
train_RMSE_lr = np.sqrt(MSE(y_train,y_train_pred_lr))
test_score_R2_lr = lr.score(X_test,y_test)
test_RMSE_lr = np.sqrt(MSE(y_test,y_test_pred_lr))
end = time.clock()
print('baseline LinearRegression train R^2:',train_score_R2_lr,'test R^2:',test_score_R2_lr
      ,'train RMSE',train_RMSE_lr,'test RMSE',test_RMSE_lr)
print('It take', end-start,'s to run')

#
from sklearn.svm import SVR
svm = SVR()
start = time.clock()
svm.fit(X_train,y_train)
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)
train_score_R2_svm = svm.score(X_train,y_train)
train_RMSE_svm = np.sqrt(MSE(y_train,y_train_pred_svm))
test_score_R2_svm = svm.score(X_test,y_test)
test_RMSE_svm = np.sqrt(MSE(y_test,y_test_pred_svm))
end = time.clock()
print('baseline SVM train R^2:',train_score_R2_svm,'test R^2:',test_score_R2_svm
      ,'train RMSE',train_RMSE_svm,'test RMSE',test_RMSE_svm)
print('It take', end-start,'s to run')
# With PCA Logistic regression Classifier vs SVM Classifier
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca.transform(X), y, test_size=.15, random_state=42)

start = time.clock()
lr.fit(X_train_pca,y_train_pca)
y_train_pred_lr = lr.predict(X_train_pca)
y_test_pred_lr = lr.predict(X_test_pca)
train_score_R2_lr = lr.score(X_train_pca,y_train_pca)
train_RMSE_lr = np.sqrt(MSE(y_train_pca,y_train_pred_lr))
test_score_R2_lr = lr.score(X_test_pca,y_test_pca)
test_RMSE_lr = np.sqrt(MSE(y_test_pca,y_test_pred_lr))
end = time.clock()
print('PCA LinearRegression train R^2:',train_score_R2_lr,'test R^2:',test_score_R2_lr
      ,'train RMSE',train_RMSE_lr,'test RMSE',test_RMSE_lr)
print('It take', end-start,'s to run')

#
start = time.clock()
svm.fit(X_train_pca,y_train_pca)
y_train_pred_svm = svm.predict(X_train_pca)
y_test_pred_svm = svm.predict(X_test_pca)
train_score_R2_svm = svm.score(X_train_pca,y_train_pca)
train_RMSE_svm = np.sqrt(MSE(y_train_pca,y_train_pred_svm))
test_score_R2_svm = svm.score(X_test_pca,y_test_pca)
test_RMSE_svm = np.sqrt(MSE(y_test_pca,y_test_pred_svm))
end = time.clock()
print('PCA SVM train R^2:',train_score_R2_svm,'test R^2:',test_score_R2_svm
      ,'train RMSE',train_RMSE_svm,'test RMSE',test_RMSE_svm)
print('It take', end-start,'s to run')


