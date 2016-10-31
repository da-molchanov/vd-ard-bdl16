# Dropout-based Automatic Relevance Determination

Here is an implimentation of new method with Automatic Relevance Determination (ARD) in linear models using variational dropout.

![](pics/nips.png)

Left: an example of an object with concatenated noise. Right: trained dropout rates, each heat map corresponds to a weight vector in multiclass model


```python
import vdrvc 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score as acc
from sklearn.cross_validation import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, t_train, t_test = train_test_split(X - np.mean(X, axis=0), y, test_size=0.2)
vd = vdrvc.vdrvc()
vd = vd.fit(X_train, t_train, num_classes=10, batch_size=1000)
```
