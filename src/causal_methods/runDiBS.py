import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import *
from dibs.utils import visualize_ground_truth
from dibs.models import ErdosReniDAGDistribution, BGe, ScaleFreeDAGDistribution, DenseNonlinearGaussian, LinearGaussian
from dibs.inference import MarginalDiBS, JointDiBS
from dibs.graph_utils import elwise_acyclic_constr_nograd
from jax.scipy.special import logsumexp
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"

# ENDOGENOUS_NODES = []

thd = 0.99
data_path = "../predictions_DriverAI_Complete/"
result_path = "../graph_all_DriverAI/"
releases = [
            'BeamNG_RF_0_7_Complete_DecisionTree',
            'BeamNG_RF_0_7_Complete_LogisticRegression',
            'BeamNG_RF_0_7_Complete_NaiveBayes',
            'BeamNG_RF_0_7_Complete_RandomForest',
            'BeamNG_RF_0_7_Complete_SVM',
            'BeamNG_RF_0_7_Complete_XGBoost',

            # 'BeamNG_RF_1_Complete_DecisionTree',
            # 'BeamNG_RF_1_Complete_LogisticRegression',
            # 'BeamNG_RF_1_Complete_NaiveBayes',
            # 'BeamNG_RF_1_Complete_RandomForest',
            # 'BeamNG_RF_1_Complete_SVM',
            # 'BeamNG_RF_1_Complete_XGBoost',
            #
            'BeamNG_RF_1_2_Complete_DecisionTree',
            'BeamNG_RF_1_2_Complete_LogisticRegression',
            'BeamNG_RF_1_2_Complete_NaiveBayes',
            'BeamNG_RF_1_2_Complete_RandomForest',
            'BeamNG_RF_1_2_Complete_SVM',
            'BeamNG_RF_1_2_Complete_XGBoost',
            #
            # 'BeamNG_RF_1_5_selected_DecisionTree',
            # 'BeamNG_RF_1_5_selected_LogisticRegression',
            # 'BeamNG_RF_1_5_selected_NaiveBayes',
            # 'BeamNG_RF_1_5_selected_RandomForest',
            # 'BeamNG_RF_1_5_selected_SVM',
            # 'BeamNG_RF_1_5_selected_XGBoost',
            #
            'BeamNG_RF_1_7_Complete_DecisionTree',
            'BeamNG_RF_1_7_Complete_LogisticRegression',
            'BeamNG_RF_1_7_Complete_NaiveBayes',
            'BeamNG_RF_1_7_Complete_RandomForest',
            'BeamNG_RF_1_7_Complete_SVM',
            'BeamNG_RF_1_7_Complete_XGBoost',
            #
            # 'BeamNG_RF_2_Complete_DecisionTree',
            # 'BeamNG_RF_2_Complete_LogisticRegression',
            # 'BeamNG_RF_2_Complete_NaiveBayes',
            # 'BeamNG_RF_2_Complete_RandomForest',
            # 'BeamNG_RF_2_Complete_SVM',
            # 'BeamNG_RF_2_Complete_XGBoost',

            # 'DriverAI_Complete_DecisionTree',
            # 'DriverAI_Complete_LogisticRegression',
            # 'DriverAI_Complete_NaiveBayes',
            # 'DriverAI_Complete_RandomForest',
            # 'DriverAI_Complete_SVM',
            # 'DriverAI_Complete_XGBoost'

            # 'BeamNG_RF_1_5_selected',
            # 'BeamNG_RF_0_7_Complete',
            # 'BeamNG_RF_1_2_Complete',
            # 'BeamNG_RF_1_7_Complete'
            ]

collected_df = pd.DataFrame()


def read_data(folder: str, selected_name:str) -> pd.DataFrame:
    df = pd.DataFrame()
    for dir_path, _, file_names in os.walk(folder):
        for file_name in file_names:
            if file_name == selected_name:
                file_path = os.path.join(folder, file_name)
                df = pd.read_csv(file_path)
                print(f"Read {file_name} with {df.shape[0]} rows")
    return df


def matrix_to_dgraph(matrix: np.ndarray, columns: List[str], threshold: float = 1.0) -> List[str]:
    dgraph = []
    # TODO:
    # for i in range(matrix.shape[0]):
    #     if matrix[i, collected_df.shape[1] - 1] > threshold:
    #         dgraph.append(f"{columns[i]} -> {columns[collected_df.shape[1] - 1]} :{matrix[i, collected_df.shape[1] - 1]}")
    #     if matrix[i, collected_df.shape[1] - 2] > threshold:
    #         dgraph.append(f"{columns[i]} -> {columns[collected_df.shape[1] - 2]} :{matrix[i, collected_df.shape[1] - 2]}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] >= threshold:
                dgraph.append(f"{columns[i]} -> {columns[j]} :{matrix[i, j]}")
    return dgraph


def compute_expected_graph(*, dist):
    n_vars = dist.g.shape[1]

    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    assert is_dag.sum() > 0,  "No acyclic graphs found"

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    expected_g = jnp.zeros_like(particles[0])
    for i in range(particles.shape[0]):
        expected_g += jnp.exp(log_weights[i]) * particles[i, :, :]

    return expected_g


def discover(release):
    selected_name = release+'.csv'
    graph_file_name = release+'_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")
    # TODO:
    ENDOGENOUS_NODES = set(collected_df.columns)

    interv_df = collected_df.copy()

    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1

    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    # model_graph = ScaleFreeDAGDistribution(collected_data.shape[1], n_edges_per_node=2)
    model_graph = ErdosReniDAGDistribution(collected_data.shape[1], n_edges_per_node=2)

    model = BGe(graph_dist=model_graph)
    dibs = MarginalDiBS(x=collected_data, interv_mask=interv_mask, inference_model=model)

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=60, steps=20000, callback_every=20000, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    # TODO:
    # visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path+graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


for release in releases:
    discover(release)

