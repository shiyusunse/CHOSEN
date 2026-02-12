# CHOSEN
Code and data for CHOSEN

## Description
Existing Test Case Prioritization (TCP) methods for ADS testing rely solely on static features from the current test
set, which may lead them to struggle to detect important features that describe the connection between
test cases and errors, thereby limiting their effectiveness in identifying critical, error-revealing tests. 

To address this limitation, we propose CHOSEN (Causality-based Hybrid-strategy mOdel SElectioN), a novel TCP
method tailored for ADS simulation testing.

### Datasets
[`/data`](/data/) stores all 7 projects.

### Repository Structure
```
CHOSEN
├─data
└─src
    ├─baselines
    │  ├─bellwether
    │  │  └─data
    │  ├─DSSDPP
    │  │  └─data
    │  └─sdc_prioritizer
    │      ├─python
    │      └─r-script
    └─causal_methods
```

### Installation
1. clone the github repository by using the following command

```
 git clone https://github.com/shiyusunse/CHOSEN.git
```
## Usage
### CHOSEN
CHOSEN consists of four parts: `Causality Discovery`, `Aggregation of Multi-Causal Analysis`, `Causal-Aided Model Performance Evaluation`, and `Hybrid-Strategy Model Selection`.

The commands below should be executed under the folder [`/src/`](/src/)
#### Causality Discovery
1. run the causal discovery methods in [`/src/causal_methods/`](/src/causal_methods/) command to get causal graphs, using BOSS as an example:

```shell
python runBOSS.py
```

2. run the command to evaluate the suspicion score for each token

```shell
python score.py
```

3. run the command to select the BoTs of top100 tokens
```shell
python select_top.py
```

4. run the command to learn the causal graph

```shell
python run_discovery.py
```

#### File-level Classfication and Global Ranking
1. run the [`main.py`](/sound/src/main.py), combining the `file-level classfication` and `global ranking`
```shell
python main.py
```
### Baselines
We provide the source code of SOTA baselines:
 | Baseline|Source path|Description|
 |---|---|---|
 |GLANCE|[`/sound/src/models/glance.py`](/sound/src/models/glance.py) | An approach that incorporates control elements and statement complexity. There are three variants based on the type of file-level classifier: GLANCE-LR, GLANCE-MD, and GLANCE-EA.|
 |LineDP|[`/sound/src/models/linedp.py`](/sound/src/models/linedp.py)|An MIT-based approach that uses a model-agnostic technique, LIME, to explain the filelevel classifier, extracting the most suspicious tokens.|
 |DeepLineDP|[`/sound/src/models/DeepLineDP_model.py`](/sound/src/models/DeepLineDP_model.py)| A deep learning approach to automatically learn the semantic properties of the surrounding tokens and lines to identify defective files/lines.|
 |N-gram|[`/sound/src/models/Ngram/n_gram.java`](/sound/src/models/Ngram/n_gram.java)| A typical natural language processing-based approach, using the entropy value to infer the naturalness of each code token.|
 |ErrorProne|[`/sound/src/models/ErrorProne/run_ErrorProne.py`](/sound/src/models/ErrorProne/run_ErrorProne.py)|A Google’s static analysis tool that builds on top of a primary Java compiler (javac) to check errors in source code based on a set of error-prone rules.|

 #### GLANCE and LineDP
 1. run the command in [`/sound/src/`](/sound/src/) to get results predicted by `GLANCE` and `LineDP` , details can be edited in `def run_default()` function
 ```shell
 python main.py
 ```

 #### DeepLineDP
 1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to modify the format of the dataset, replacing " with "" to accommodate the requirements of DeepLineDP's code and generalize the dataset in both `File-level` and `Line-level`

 ```shell
python preprocess_file_dataset.py
python preprocess_line_dataset.py
 ```

2. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for file-level model training

```shell
python preprocess_data.py
```

3. run the command in [`/sound/src/models/`](/sound/src/models/) to train Word2Vec models

```shell
python train_word2vec.py
```

4. run the command in [`/sound/src/models/`](/sound/src/models/) to train DeepLineDP models

```shell
python train_model.py
```

5. run the command in [`/sound/src/`](/sound/src/) to make a prediction of each software release

```shell
python generate_prediction.py
```

6. run the command in [`/sound/src/`](/sound/src/) to get the result of experiments

```
Rscript  get_evaluation_result.R
```

#### N-gram
1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for `N-gram` and `ErrorProne`

```
python export_data_for_line_level_baseline.py
```

2. run the command in [`/sound/src/models/Ngram/`](/sound/src/models/Ngram/) to obtain results from `N-gram`

```shell
javac -cp .:slp-core.jar:commons-io-2.8.0.jar n_gram.java
java -cp .:slp-core.jar:commons-io-2.8.0.jar n_gram
```

#### ErrorProne
1. run the command in [`/sound/src/tools/`](/sound/src/tools/) to prepare data for `N-gram` and `ErrorProne`

```
python export_data_for_line_level_baseline.py
```
2. run the command in [`/sound/src/models/ErrorProne/`](/sound/src/models/ErrorProne/) to get rsults from `ErrorProne`

```shell
python run_ErrorProne.py
```
### RQs
The data and figures for RQs can be gennerated by files in [`/sound/exp/`](/sound/exp/)
