from dataset.cwru.rauber_loca_et_al import single_channel_X_y_DE_FE_12k
from dataset.cwru.sehri_et_al import single_channel_X_y_DE_FE_48k
from assesment.crossvalidation import performance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from estimator.WPRF import WPRF as Estimator


model = Estimator()

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
list_of_metrics = [accuracy_score, f1_macro, confusion_matrix]


def run(model, verbose=False):
    combination = 0
    segment_length = 2048
    list_of_X_y = single_channel_X_y_DE_FE_48k(combination, segment_length)
    scores = performance(model, list_of_X_y, list_of_metrics=list_of_metrics, 
                         holdout_indices=[[0, 1, 2, 3, 4, 5, 6, 7], [8]], verbose=verbose)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vibrolearn with the following options:")
    parser.add_argument("-a", "--augmentation", action="store_true", help="Whether to use data augmentation in the experiments")
    parser.add_argument("-f", "--feature_extraction", type=str, help="The feature extraction method to use for the experiments (choices: FlattenFeatures, StatisticalFeatures, HeterogeneousFeatures, WaveletFeatures)")
    parser.add_argument("-c", "--classifier", type=str, help="The classifier to use for the experiments (choices: RandomForestClassifier, KNeighborsClassifier)")
    parser.add_argument("-e", "--experimental_setup", type=str, help="The experimental setup file to run (mandatory)")


    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()


    steps = []


    featextraction_method = WaveletFeatures()
    if args.feature_extraction:
        featextraction_method = eval(args.feature_extraction)()
    print(f"Using feature extraction method: {featextraction_method.__class__.__name__}")
    steps.append(("feature_extraction", featextraction_method))


    model = RandomForestClassifier()
    if args.classifier:
        model = eval(args.classifier)()
    print(f"Using classifier: {model.__class__.__name__}")
    steps.append(("classifier", model))


    pipe = Pipeline(steps)


    if args.augmentation:
        print("Using data augmentation")
        pipe.train_loader = augmented


    if args.experimental_setup:
        print(f"Running experimental setup: {args.experimental_setup}")
        save_experiment_results(args, pipe)
    elif not args.experimental_setup:
        print("No experimental setup specified, please provide one using the -e or --experimental_setup argument")

