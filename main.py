from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, mutual_info_classif, f_classif, SelectPercentile, VarianceThreshold, SelectFromModel, RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from feature.extraction import WaveletFeatures
from feature.wp import WPDFeatureSelector
from utils.metrics import f1_macro
from feature.features_selector import (WPDFeatureSelectorMI, WPDFeatureSelectorANOVA, BaseFeatureSelector, 
WPDFeatureSelectorRFStability, WPDFeatureSelectorCorrMI, WPDFeatureSelectorPermutation, WPDFeatureSelectorRFImportance)
model = make_pipeline(WaveletFeatures(), RandomForestClassifier())

# model = make_pipeline(
#     WaveletFeatures(),
#     SelectKBest(f_classif, k=8),
#     RandomForestClassifier(n_estimators=200, random_state=0)
# )
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

""" CWRU without augmentation """
# def run_sehri_experiment():
#     from dataset.cwru.sehri_et_al import run_papers_experiment
#     scores = run_papers_experiment(model, list_of_metrics)
#     print("Scores for papers experiment:")
#     pprint(scores)

""" CWRU with augmentation """
def run_sehri_experiment_aug():
    from dataset.cwru.sehri_et_al_aug import run_papers_experiment
    scores = run_papers_experiment(model, list_of_metrics, mixes_per_pair=5, augment=True)
    print("Scores for papers experiment:")
    pprint(scores)
    return scores

""" CWRU with augmentation and CNN1D """
# def run_sehri_experiment_aug():
#     from dataset.cwru.sehri_et_al_aug import run_papers_experiment    
#     from estimator.cnn1d import SklearnCNN1DClassifier
#     model = SklearnCNN1DClassifier(
#         epochs=20,
#         batch_size=64,
#         lr=1e-4,
#         seed=0,
#         verbose=True,
#     )
#     scores = run_papers_experiment(model, list_of_metrics, mixes_per_pair=5, augment=True)
#     print("Scores for papers experiment:")
#     pprint(scores)
#     return scores

""" UORED """
# def run_sehri_experiment_aug():
#     from dataset.uored.experimenter import run_papers_experiment
#     scores = run_papers_experiment(
#         model,
#         list_of_metrics,
#         mixes_per_pair=1,
#         augment=True,
#         channel="vibration",   # or "acoustic"
#     )
#     print("Scores for papers experiment:")
#     pprint(scores)
#     return scores

""" Papers inspired experiment """
# def run_inspired_experiment():
#     from dataset.cwru.sehri_et_al import run_papers_inspired_experiment
#     scores = run_papers_inspired_experiment(model, list_of_metrics)
#     print("Scores for papers inspired experiment:")
#     pprint(scores)

""" Proposed experiment """
# def run_proposed_experiment():
#     from dataset.cwru.sehri_et_al import run_proposed_experiment
#     scores = run_proposed_experiment(model, list_of_metrics)
#     print("Scores for proposed experiment:")
#     pprint(scores)

if __name__ == "__main__":
    accuracies = []
    f1_macros = []
    for i in range(1):
        result = run_sehri_experiment_aug()
        accuracies.append(result['accuracy_score'])
        f1_macros.append(result['f1_macro'])
    print("Average accuracy over 5 runs:", sum(accuracies) / len(accuracies))
    print("Average f1_macro over 5 runs:", sum(f1_macros) / len(f1_macros))
