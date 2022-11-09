# GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection

This is the source code of WSDM'23 paper "GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection".

## Requirements
This code requires the following:
* Python==3.9
* Pytorch==1.11.0
* Pytorch Geometric==2.0.4
* Numpy==1.21.2
* Scikit-learn==1.0.2
* OGB==1.3.3
* NetworkX==2.7.1
* FAISS-GPU==1.7.2

## Usage
Just run the script corresponding to the experiment and dataset you want. For instance:

* Run out-of-distribution detection on BZR (ID) and COX2 (OOD) datasets:
```
bash script/oodd_BZR+COX2.sh
```

* Run anomaly detection on PROTEINS_full datasets:
```
bash script/ad_PROTEINS_full.sh
```

## Statistic of Graph-level OOD Detection Benchmark

The statistic of each dataset pair in our benchmark is provided as follows.

<table>
  <tr>
  <td> </td><td colspan="4">ID dataset</td><td colspan="4">OOD dataset</td>
  </tr>
  <tr><td>No.</td><td>Name</td><td># Graph<br>(Train/Test)</td><td># Node<br>(avg.)</td><td># Edge<br>(avg.)</td>
                  <td>Name</td><td># Graph<br>(Test)</td><td># Node<br>(avg.)</td><td># Edge<br>(avg.)</td>
  </tr>
  <tr><td>1</td><td>BZR</td><td>364/41</td><td>35.8</td><td>38.4</td>
                <td>COX2</td><td>41</td><td>41.2</td><td>43.5</td>
  </tr>
  <tr><td>2</td><td>PTC-MR</td><td>309/35</td><td>14.3</td><td>14.7</td>
                <td>MUTAG</td><td>35</td><td>17.9</td><td>19.8</td>
  </tr>
  <tr><td>3</td><td>AIDS</td><td>1,800/200</td><td>15.7</td><td>16.2</td>
                <td>DHFR</td><td>200</td><td>42.4</td><td>44.5</td>
  </tr>
  <tr><td>4</td><td>ENZYMES</td><td>540/60</td><td>32.6</td><td>62.1</td>
                <td>PROTEIN</td><td>60</td><td>39.1</td><td>72.8</td>
  </tr>
  <tr><td>5</td><td>IMDB-B</td><td>1,350/150</td><td>19.8</td><td>96.5</td>
                <td>IMDB-M</td><td>150</td><td>13.0</td><td>65.9</td>
  </tr>
  <tr><td>6</td><td>Tox21</td><td>7,047/784</td><td>18.6</td><td>19.3</td>
                <td>SIDER</td><td>784</td><td>33.6</td><td>35.4</td>
  </tr>
  <tr><td>7</td><td>FreeSolv</td><td>577/65</td><td>8.7</td><td>8.4</td>
                <td>ToxCast</td><td>65</td><td>18.8</td><td>19.3</td>
  </tr>
  <tr><td>8</td><td>BBBP</td><td>1,835/204</td><td>24.1</td><td>26.0</td>
                <td>BACE</td><td>204</td><td>34.1</td><td>36.9</td>
  </tr>
  <tr><td>9</td><td>ClinTox</td><td>1,329/148</td><td>26.2</td><td>27.9</td>
                <td>LIPO</td><td>148</td><td>27.0</td><td>29.5</td>
  </tr>
  <tr><td>10</td><td>Esol</td><td>1,015/113</td><td>13.3</td><td>13.7</td>
                <td>MUV</td><td>113</td><td>24.2</td><td>26.3</td>
  </tr>
</table>

## Statistic of Graph-level Anomaly Detection Datasets

The statistic of each dataset in the anomaly detection experiments is provided as follows.

<table>
  <tr><td>Dataset</td><td># Graph<br>(Train/Test)</td><td># Node<br>(avg.)</td><td># Edge<br>(avg.)</td></tr>
  <tr><td>PROTEINS-full</td><td>360/223</td><td>39.1</td><td>72.8</td></tr>
  <tr><td>ENZYMES</td><td>400/120</td><td>32.6</td><td>62.1</td></tr>
  <tr><td>AIDS</td><td>1280/400</td><td>15.7</td><td>16.2</td></tr>
  <tr><td>DHFR</td><td>368/152</td><td>42.4</td><td>44.5</td></tr>
  <tr><td>BZR</td><td>69/81</td><td>35.8</td><td>38.4</td></tr>
  <tr><td>COX2</td><td>81/94</td><td>41.2</td><td>43.5</td></tr>
  <tr><td>DD</td><td>390/236</td><td>284.3</td><td>715.7</td></tr>
  <tr><td>NCI1</td><td>1646/822</td><td>29.8</td><td>32.3</td></tr>
  <tr><td>IMDB-B</td><td>400/200</td><td>19.8</td><td>96.5</td></tr>
  <tr><td>REDDIT-B</td><td>800/400</td><td>429.6</td><td>497.8</td></tr>
  <tr><td>COLLAB</td><td>1920/1000</td><td>74.5</td><td>2457.8</td></tr>
  <tr><td>HSE</td><td>423/267</td><td>16.9</td><td>17.2</td></tr>
  <tr><td>MMP</td><td>6170/238</td><td>17.6</td><td>18.0</td></tr>
  <tr><td>p53</td><td>8088/269</td><td>17.9</td><td>18.3</td></tr>
  <tr><td>PPAR-gamma</td><td>219/267</td><td>17.4</td><td>17.7</td></tr>
</table>


## Implementation Details

### Hyper-parameters

For the sake of efficiency, we set the structural encoding dimensions $d_s^{(rw)}$ and $d_s^{(dg)}$ to $16$. The encoders are 5-layer GINs with $16$ hidden dimensions. The number of dimensions of projected embeddings is the same as which of node embeddings. The batch size is selected from $16$ to $128$ according to the graph size of datasets. The number of clusters $K$ and self-adaptiveness parameter $\alpha$ are selected through grid search, with the scopes of $\{2, 3, 5, 10, 15, 20, 30\}$ and $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$, respectively. The model is trained by the Adam optimizer with a learning rate of $0.0001$ until converging. 

### Computing Infrastructures

We conduct the experiments on a Linux server with an Intel Xeon Gold 6226R CPU and two Tesla V100S GPUs. We implement our method with PyTorch 1.11.0 and Pytorch Geometric 2.0.4. 

## Cite

If you compare with, build on, or use aspects of this work, please cite the following:
```
@inproceedings{liu2023goodd,
  title={GOOD-D: On Unsupervised Graph Out-Of-Distribution Detection},
  author={Liu, Yixin and Ding, Kaize and Liu, Huan and Pan, Shirui},
  booktitle={Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining},
  year={2023}
}
```
