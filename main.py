<<<<<<< HEAD
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from feature.extraction import WaveletFeatures
from utils.metrics import f1_macro
=======
from dataset.cwru.rauber_loca_et_al import single_channel_X_y_DE_FE_12k
from dataset.cwru.sehri_et_al import single_channel_X_y_DE_FE_48k
from assesment.crossvalidation import performance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from estimator.WPRF import WPRF as Estimator

>>>>>>> 0b48f41d6b0eea3352b7411cdac0b62d995b3371

model = make_pipeline(WaveletFeatures(), RandomForestClassifier())

list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]

def run_sehri_experiment():
    from dataset.cwru.sehri_et_al import run_papers_experiment
    scores = run_papers_experiment(model, list_of_metrics)
    print("Scores for papers experiment:")
    pprint(scores)

def run_inspired_experiment():
    from dataset.cwru.sehri_et_al import run_papers_inspired_experiment
    scores = run_papers_inspired_experiment(model, list_of_metrics)
    print("Scores for papers inspired experiment:")
    pprint(scores)

def run_proposed_experiment():
    from dataset.cwru.sehri_et_al import run_proposed_experiment
    scores = run_proposed_experiment(model, list_of_metrics)
    print("Scores for proposed experiment:")
    pprint(scores)

def run(model, verbose=False):
    combination = 0
    segment_length = 2048
    list_of_X_y = single_channel_X_y_DE_FE_48k(combination, segment_length)
    scores = performance(model, list_of_X_y, list_of_metrics=list_of_metrics, verbose=verbose)
    return scores


if __name__ == "__main__":
<<<<<<< HEAD
    run_sehri_experiment()
    # run_inspired_experiment()
    # run_proposed_experiment()
=======
    result  = run(model, verbose=True)
>>>>>>> 0b48f41d6b0eea3352b7411cdac0b62d995b3371
