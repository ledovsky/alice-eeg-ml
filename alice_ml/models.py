import importlib.resources as pkg_resources
import joblib

import pandas as pd

from alice_ml.features import get_features_from_mne
from alice_ml import pretrained


def predict_mne(raw, ica):
    
    features_df = get_features_from_mne(raw, ica)
    flags = ['flag_brain', 'flag_alpha', 'flag_mu', 'flag_muscles', 'flag_eyes', 'flag_heart', 'flag_ch_noise']

    scaler = joblib.load(pkg_resources.open_binary(pretrained, 'scaler.joblib'))

    models = {}
    for flag in flags:
        models[flag] = joblib.load(pkg_resources.open_binary(pretrained, 'lr_' + flag + '.joblib'))
    
    X = scaler.transform(features_df)
    cols = {}
    for flag in flags:
        cols[flag] = models[flag].predict_proba(X)[:, 1]
    
    pred_df = pd.DataFrame(cols, index=features_df.index)
    return pred_df
    