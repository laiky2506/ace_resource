### Link to notebooks
* Task 1: [https://example.com](https://example.com)
* Task 2: [https://example.com](https://example.com)
* Task 3: [https://example.com](https://example.com)

# QUESTION 1
## Task 1a: 
A descriptive analysis of the additives (columns named as “a” to “i”), which must include summaries of findings (parametric/non-parametric). Correlation and ANOVA, if applicable, is a must.

### Introduction
<p> This question is solved using Jupyter Notebooks, then uploaded to XXX. The solution steps as follow: </p>
* Import modules
* Load data into pandas DataFrame
* Use describe() method to generate a descriptive information of the dataset
* Run Shapiro-Wilk test to check whether to run a parametric or non parametric test
* From the test above, non-parametric test should be carried out. Spearman's Rank Correlation is used to do the correlation
* Kruskal Wallis H Test is carried out as alternative to ANOVA

### Step 1: Modules Used For This Project
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### STEP 2: Read CSV As DataFrame And Generate Descripive Information Of The Dataset
```
data = pd.read_csv(r"https://docs.google.com/spreadsheets/d/e/2PACX-1vQOdsaFChbSVH7QXsEOLLJNZiL3lr5uFg8ZvBA3tTHKNreaPwZvTA3WQN4LN5f_vYgX_TxkpZKOt0l9/pub?output=csv")
data.describe()
```

![Image](img/q1/img_001.jpg)



since the assumption of normality for ANOVA aren't met, 
