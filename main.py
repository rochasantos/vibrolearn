import argparse
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from estimators.pipeline import Pipeline
from experiment.assesment import print_scores_list, run_experiment
from feature.extraction import *
from preprocessing.augmentation import AugmentedPipeline


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
    pipeline = Pipeline

    if args.augmentation:
        print("Using data augmentation")
        pipeline = AugmentedPipeline


    if args.feature_extraction:
        print(f"Using feature extraction method: {args.feature_extraction}")
        featextraction_method = eval(args.feature_extraction)()
        steps.append(("feature_extraction", featextraction_method))
    elif not args.feature_extraction:
        print("No feature extraction method specified, using default (WaveletFeatures)")
        featextraction_method = WaveletFeatures()
        steps.append(("feature_extraction", featextraction_method))
        
    if args.classifier:
        model = eval(args.classifier)()
        print(f"Using classifier: {model.__class__.__name__}")
        steps.append(("classifier", model))
    elif not args.classifier:
        model = RandomForestClassifier()
        print("No classifier specified, using default (RandomForestClassifier)")
        steps.append(("classifier", model))

    if args.experimental_setup:
        print(f"Running experimental setup: {args.experimental_setup}")
        pipe = pipeline(steps)
        experimental_setup = json.load(open(args.experimental_setup, "r"))
        list_of_scores = run_experiment(pipe, experimental_setup)
        print_scores_list(list_of_scores)
    elif not args.experimental_setup:
        print("No experimental setup specified, please provide one using the -e or --experimental_setup argument")

