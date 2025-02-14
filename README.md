## Importing the Dependencies


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Data Collection and Data Processing


```python
# loading the dataset to a pandas dataframe
sonar_data=pd.read_csv(r"Sonar_data.csv",header=None)
```


```python
sonar_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>




```python
sonar_data.shape
```




    (208, 61)




```python
sonar_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>...</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029164</td>
      <td>0.038437</td>
      <td>0.043832</td>
      <td>0.053892</td>
      <td>0.075202</td>
      <td>0.104570</td>
      <td>0.121747</td>
      <td>0.134799</td>
      <td>0.178003</td>
      <td>0.208259</td>
      <td>...</td>
      <td>0.016069</td>
      <td>0.013420</td>
      <td>0.010709</td>
      <td>0.010941</td>
      <td>0.009290</td>
      <td>0.008222</td>
      <td>0.007820</td>
      <td>0.007949</td>
      <td>0.007941</td>
      <td>0.006507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022991</td>
      <td>0.032960</td>
      <td>0.038428</td>
      <td>0.046528</td>
      <td>0.055552</td>
      <td>0.059105</td>
      <td>0.061788</td>
      <td>0.085152</td>
      <td>0.118387</td>
      <td>0.134416</td>
      <td>...</td>
      <td>0.012008</td>
      <td>0.009634</td>
      <td>0.007060</td>
      <td>0.007301</td>
      <td>0.007088</td>
      <td>0.005736</td>
      <td>0.005785</td>
      <td>0.006470</td>
      <td>0.006181</td>
      <td>0.005031</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001500</td>
      <td>0.000600</td>
      <td>0.001500</td>
      <td>0.005800</td>
      <td>0.006700</td>
      <td>0.010200</td>
      <td>0.003300</td>
      <td>0.005500</td>
      <td>0.007500</td>
      <td>0.011300</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000800</td>
      <td>0.000500</td>
      <td>0.001000</td>
      <td>0.000600</td>
      <td>0.000400</td>
      <td>0.000300</td>
      <td>0.000300</td>
      <td>0.000100</td>
      <td>0.000600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.013350</td>
      <td>0.016450</td>
      <td>0.018950</td>
      <td>0.024375</td>
      <td>0.038050</td>
      <td>0.067025</td>
      <td>0.080900</td>
      <td>0.080425</td>
      <td>0.097025</td>
      <td>0.111275</td>
      <td>...</td>
      <td>0.008425</td>
      <td>0.007275</td>
      <td>0.005075</td>
      <td>0.005375</td>
      <td>0.004150</td>
      <td>0.004400</td>
      <td>0.003700</td>
      <td>0.003600</td>
      <td>0.003675</td>
      <td>0.003100</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.022800</td>
      <td>0.030800</td>
      <td>0.034300</td>
      <td>0.044050</td>
      <td>0.062500</td>
      <td>0.092150</td>
      <td>0.106950</td>
      <td>0.112100</td>
      <td>0.152250</td>
      <td>0.182400</td>
      <td>...</td>
      <td>0.013900</td>
      <td>0.011400</td>
      <td>0.009550</td>
      <td>0.009300</td>
      <td>0.007500</td>
      <td>0.006850</td>
      <td>0.005950</td>
      <td>0.005800</td>
      <td>0.006400</td>
      <td>0.005300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.035550</td>
      <td>0.047950</td>
      <td>0.057950</td>
      <td>0.064500</td>
      <td>0.100275</td>
      <td>0.134125</td>
      <td>0.154000</td>
      <td>0.169600</td>
      <td>0.233425</td>
      <td>0.268700</td>
      <td>...</td>
      <td>0.020825</td>
      <td>0.016725</td>
      <td>0.014900</td>
      <td>0.014500</td>
      <td>0.012100</td>
      <td>0.010575</td>
      <td>0.010425</td>
      <td>0.010350</td>
      <td>0.010325</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.137100</td>
      <td>0.233900</td>
      <td>0.305900</td>
      <td>0.426400</td>
      <td>0.401000</td>
      <td>0.382300</td>
      <td>0.372900</td>
      <td>0.459000</td>
      <td>0.682800</td>
      <td>0.710600</td>
      <td>...</td>
      <td>0.100400</td>
      <td>0.070900</td>
      <td>0.039000</td>
      <td>0.035200</td>
      <td>0.044700</td>
      <td>0.039400</td>
      <td>0.035500</td>
      <td>0.044000</td>
      <td>0.036400</td>
      <td>0.043900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>




```python
sonar_data[60].value_counts()
```




    60
    M    111
    R     97
    Name: count, dtype: int64



M--> Mine
R--> Rock


```python
sonar_data.groupby(60).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
    <tr>
      <th>60</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>0.034989</td>
      <td>0.045544</td>
      <td>0.050720</td>
      <td>0.064768</td>
      <td>0.086715</td>
      <td>0.111864</td>
      <td>0.128359</td>
      <td>0.149832</td>
      <td>0.213492</td>
      <td>0.251022</td>
      <td>...</td>
      <td>0.019352</td>
      <td>0.016014</td>
      <td>0.011643</td>
      <td>0.012185</td>
      <td>0.009923</td>
      <td>0.008914</td>
      <td>0.007825</td>
      <td>0.009060</td>
      <td>0.008695</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>R</th>
      <td>0.022498</td>
      <td>0.030303</td>
      <td>0.035951</td>
      <td>0.041447</td>
      <td>0.062028</td>
      <td>0.096224</td>
      <td>0.114180</td>
      <td>0.117596</td>
      <td>0.137392</td>
      <td>0.159325</td>
      <td>...</td>
      <td>0.012311</td>
      <td>0.010453</td>
      <td>0.009640</td>
      <td>0.009518</td>
      <td>0.008567</td>
      <td>0.007430</td>
      <td>0.007814</td>
      <td>0.006677</td>
      <td>0.007078</td>
      <td>0.006024</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 60 columns</p>
</div>




```python
# Separating Data and Labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
```


```python
print(X)
print(Y)
```

             0       1       2       3       4       5       6       7       8   \
    0    0.0200  0.0371  0.0428  0.0207  0.0954  0.0986  0.1539  0.1601  0.3109   
    1    0.0453  0.0523  0.0843  0.0689  0.1183  0.2583  0.2156  0.3481  0.3337   
    2    0.0262  0.0582  0.1099  0.1083  0.0974  0.2280  0.2431  0.3771  0.5598   
    3    0.0100  0.0171  0.0623  0.0205  0.0205  0.0368  0.1098  0.1276  0.0598   
    4    0.0762  0.0666  0.0481  0.0394  0.0590  0.0649  0.1209  0.2467  0.3564   
    ..      ...     ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.0187  0.0346  0.0168  0.0177  0.0393  0.1630  0.2028  0.1694  0.2328   
    204  0.0323  0.0101  0.0298  0.0564  0.0760  0.0958  0.0990  0.1018  0.1030   
    205  0.0522  0.0437  0.0180  0.0292  0.0351  0.1171  0.1257  0.1178  0.1258   
    206  0.0303  0.0353  0.0490  0.0608  0.0167  0.1354  0.1465  0.1123  0.1945   
    207  0.0260  0.0363  0.0136  0.0272  0.0214  0.0338  0.0655  0.1400  0.1843   
    
             9   ...      50      51      52      53      54      55      56  \
    0    0.2111  ...  0.0232  0.0027  0.0065  0.0159  0.0072  0.0167  0.0180   
    1    0.2872  ...  0.0125  0.0084  0.0089  0.0048  0.0094  0.0191  0.0140   
    2    0.6194  ...  0.0033  0.0232  0.0166  0.0095  0.0180  0.0244  0.0316   
    3    0.1264  ...  0.0241  0.0121  0.0036  0.0150  0.0085  0.0073  0.0050   
    4    0.4459  ...  0.0156  0.0031  0.0054  0.0105  0.0110  0.0015  0.0072   
    ..      ...  ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.2684  ...  0.0203  0.0116  0.0098  0.0199  0.0033  0.0101  0.0065   
    204  0.2154  ...  0.0051  0.0061  0.0093  0.0135  0.0063  0.0063  0.0034   
    205  0.2529  ...  0.0155  0.0160  0.0029  0.0051  0.0062  0.0089  0.0140   
    206  0.2354  ...  0.0042  0.0086  0.0046  0.0126  0.0036  0.0035  0.0034   
    207  0.2354  ...  0.0181  0.0146  0.0129  0.0047  0.0039  0.0061  0.0040   
    
             57      58      59  
    0    0.0084  0.0090  0.0032  
    1    0.0049  0.0052  0.0044  
    2    0.0164  0.0095  0.0078  
    3    0.0044  0.0040  0.0117  
    4    0.0048  0.0107  0.0094  
    ..      ...     ...     ...  
    203  0.0115  0.0193  0.0157  
    204  0.0032  0.0062  0.0067  
    205  0.0138  0.0077  0.0031  
    206  0.0079  0.0036  0.0048  
    207  0.0036  0.0061  0.0115  
    
    [208 rows x 60 columns]
    0      R
    1      R
    2      R
    3      R
    4      R
          ..
    203    M
    204    M
    205    M
    206    M
    207    M
    Name: 60, Length: 208, dtype: object
    

## Training and Test Data


```python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
```


```python
print(X.shape,X_train.shape,X_test.shape)
```

    (208, 60) (187, 60) (21, 60)
    


```python
print(X_train)
print(Y_train)
```

             0       1       2       3       4       5       6       7       8   \
    115  0.0414  0.0436  0.0447  0.0844  0.0419  0.1215  0.2002  0.1516  0.0818   
    38   0.0123  0.0022  0.0196  0.0206  0.0180  0.0492  0.0033  0.0398  0.0791   
    56   0.0152  0.0102  0.0113  0.0263  0.0097  0.0391  0.0857  0.0915  0.0949   
    123  0.0270  0.0163  0.0341  0.0247  0.0822  0.1256  0.1323  0.1584  0.2017   
    18   0.0270  0.0092  0.0145  0.0278  0.0412  0.0757  0.1026  0.1138  0.0794   
    ..      ...     ...     ...     ...     ...     ...     ...     ...     ...   
    140  0.0412  0.1135  0.0518  0.0232  0.0646  0.1124  0.1787  0.2407  0.2682   
    5    0.0286  0.0453  0.0277  0.0174  0.0384  0.0990  0.1201  0.1833  0.2105   
    154  0.0117  0.0069  0.0279  0.0583  0.0915  0.1267  0.1577  0.1927  0.2361   
    131  0.1150  0.1163  0.0866  0.0358  0.0232  0.1267  0.2417  0.2661  0.4346   
    203  0.0187  0.0346  0.0168  0.0177  0.0393  0.1630  0.2028  0.1694  0.2328   
    
             9   ...      50      51      52      53      54      55      56  \
    115  0.1975  ...  0.0222  0.0045  0.0136  0.0113  0.0053  0.0165  0.0141   
    38   0.0475  ...  0.0149  0.0125  0.0134  0.0026  0.0038  0.0018  0.0113   
    56   0.1504  ...  0.0048  0.0049  0.0041  0.0036  0.0013  0.0046  0.0037   
    123  0.2122  ...  0.0197  0.0189  0.0204  0.0085  0.0043  0.0092  0.0138   
    18   0.1520  ...  0.0045  0.0084  0.0010  0.0018  0.0068  0.0039  0.0120   
    ..      ...  ...     ...     ...     ...     ...     ...     ...     ...   
    140  0.2058  ...  0.0798  0.0376  0.0143  0.0272  0.0127  0.0166  0.0095   
    5    0.3039  ...  0.0104  0.0045  0.0014  0.0038  0.0013  0.0089  0.0057   
    154  0.2169  ...  0.0039  0.0053  0.0029  0.0020  0.0013  0.0029  0.0020   
    131  0.5378  ...  0.0228  0.0099  0.0065  0.0085  0.0166  0.0110  0.0190   
    203  0.2684  ...  0.0203  0.0116  0.0098  0.0199  0.0033  0.0101  0.0065   
    
             57      58      59  
    115  0.0077  0.0246  0.0198  
    38   0.0058  0.0047  0.0071  
    56   0.0011  0.0034  0.0033  
    123  0.0094  0.0105  0.0093  
    18   0.0132  0.0070  0.0088  
    ..      ...     ...     ...  
    140  0.0225  0.0098  0.0085  
    5    0.0027  0.0051  0.0062  
    154  0.0062  0.0026  0.0052  
    131  0.0141  0.0068  0.0086  
    203  0.0115  0.0193  0.0157  
    
    [187 rows x 60 columns]
    115    M
    38     R
    56     R
    123    M
    18     R
          ..
    140    M
    5      R
    154    M
    131    M
    203    M
    Name: 60, Length: 187, dtype: object
    

Model Training--> Logistic Regression


```python
model=LogisticRegression()
```


```python
model.fit(X_train,Y_train)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-4 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>LogisticRegression</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression()</pre></div> </div></div></div></div>



## Model Evaluation


```python
# accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
```


```python
print("Accuracy on training data :",training_data_accuracy)
```

    Accuracy on training data : 0.8342245989304813
    


```python
# accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
```


```python
print("Accuracy on test data :",test_data_accuracy)
```

    Accuracy on test data : 0.7619047619047619
    

## Making a predictive dataset


```python
input_data=(0.0286,0.0453,0.0277,0.0174,0.0384,0.099,0.1201,0.1833,0.2105,0.3039,0.2988,0.425,0.6343,0.8198,1,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.589,0.2872,0.2043,0.5782,0.5389,0.375,0.3411,0.5067,0.558,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.265,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

#changing the input_data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if prediction[0]=='R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

    ['R']
    The object is a Rock
    
