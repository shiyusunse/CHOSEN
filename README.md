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
#### CHOSEN
1. run the [`main.py`](/src/main.py), with the causal graphs discovered.
```shell
python main.py
```
### Baselines
We provide the source code of SOTA baselines:
 | Baseline|Source path|Description|
 |---|---|---|
 |SO-SDC-Prioritizer|[`/src/baselines/sdc_prioritizer/main.m`](/src/baselines/sdc_prioritizer/main.m) | SO-SDC-Prioritizer is a Single-Objective optimization-based test case prioritization approach. The fitness function balances test diversity and execution cost, aiming to maximize diversity while minimizing execution time|
 |MO-SDC-Prioritizer|[`/src/baselines/sdc_prioritizer/main.m`](/sound/src/models/linedp.py)| MO-SDC-Prioritizer extends SO-SDC-Prioritizer into a Multi-Objective optimization framework by adopting NSGA-II to simultaneously optimize two conflicting objectives, i.e., maximizing test diversity and minimizing execution cost.|
 |Greedy|[`/src/baselines/sdc_prioritizer/mainGreedy.m`](/src/baselines/sdc_prioritizer/mainGreedy.m)|Greedy is a black-box prioritization technique that orders test execution using a cost-effectiveness heuristic.|
 |DSSDPP|[`/src/baselines/DSSDPP/main_DSSDPP_add_indicators.py`](/src/baselines/DSSDPP/main_DSSDPP_add_indicators.py)|DSSDPP, short for Data Selection and Sampling based Domain Programming Predictor, is a non-parametric transfer learning approach for cross-project defect prediction (CPDP).|
 | Bellwether|[`/sound/src/models/ErrorProne/run_ErrorProne.py`](/src/baselines/bellwether/run_two_stage_bellwether_add_indicators.py)|Bellwether is a widely-used transfer learning approach for CPDP. The key idea is that within a collection of software projects, there is often a single project—the bellwether project—whose data can serve as a reliable and representative source for predicting defects in other projects.|
### RQ & Discussion
The data and figures for RQs can be gennerated by files in [`/src/exp/`](/src/exp/)
