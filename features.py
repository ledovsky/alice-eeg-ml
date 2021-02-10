import numpy as np

from utils import *

eye_move_example = np.load('eye_move_example.npy')
eye_blink_example = np.load('eye_blink_example.npy')


def _cross_corr(epoch_eye, epoch):
    ccov = np.correlate(epoch_eye - epoch_eye.mean(), epoch - epoch.mean(), mode='same')
    ccor = ccov / (len(epoch_eye) * epoch_eye.std() * epoch.std())

    return ccor


def compute_K(ic, mean_shift=False, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        mean_shift (bool, optional): if set to True, the average over epoch is subtracted. Defaults to False.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Temporal Kurtosis feature.
    """
    def _epoch_kurtosis(epoch):
        if not mean_shift:
            return np.mean(epoch ** 4) / (np.mean(epoch ** 2) ** 2) - 3
        return np.mean((epoch - epoch.mean()) ** 4) / (epoch.var() ** 2) - 3

    epochs = ic.signal.apply(_epoch_kurtosis)

    if not thres:
        return epochs.mean()
    return trim(epochs, thres).mean()


def compute_MEV(ic, thres=0.99) -> float:
    """
    Args:
        ic (IC): indepentent component.
        thres (float, optional): determines whether to remove the top 1 - thres of values. Defaults to 0.99.

    Returns:
        float: Maximum Epoch Variance feature.
    """
    vars = ic.signal.apply(np.var)

    if not thres:
        return vars.max() / vars.mean()
    return vars.quantile(thres) / trim(vars, thres).mean()


FA = {'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
      'af3', 'af4', 'af7', 'af8',
      'fp1', 'fpz', 'fp2'}
PA = {'cpz', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5', 'cp6',
      'pz', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 't5', 't6', 'p9', 'p10',
      'poz', 'po3', 'po4', 'po7', 'po8',
      'oz', 'o1', 'o2'}
LE = {'f3', 'f5', 'f7', 'f9',
      'af7'}
RE = {'f4', 'f6', 'f8', 'f10',
      'af6'}

FC = {'fp1','fpz','fp2','f3','fz','f4','fc3','fcz','fc4','c3','cz','c4'}
FC_not={'f7','ft7','t3','tp7','t5','t6','tp8','t4','ft8','f8','cp3','cpz','cp4','p3','pz','p4','o1','oz','o2'}
FP= {'c3','cz','c4','cp3','cpz','cp4','p3','pz','p4','o1','oz','o2'}
FP_not={'f7','ft7','t3','tp7','t5','t6','tp8','t4','ft8','f8','fp1','fpz','fp2','f3','fz','f4','fc3','fcz','fc4'}


def compute_AT(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Alpha Topography - Difference between weights in frontal-central sites and other sites
    """
    return np.abs(ic.select_weights(FC).mean()) - np.abs(ic.select_weights(FC_not).mean())

def compute_MT(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Mu Topography - Difference between weights in occipital-parietal sites and other sites
    """
    return np.abs(ic.select_weights(FP).mean()) - np.abs(ic.select_weights(FP_not).mean())

def compute_AMALB(ic)  -> float:
    
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Average Magnitude in ALpha Band
    """


    freqs, psd=ic.psd(verbose=False)
    
    alphs_inds=np.where((freqs>=6)&(freqs<=12))


    alphs_inds_not=np.where((freqs<6)|(freqs>12))
    
    mean_psd = psd.mean(axis=0)
    
    return np.mean(mean_psd[alphs_inds])/np.mean(mean_psd[alphs_inds_not])
    
    



def compute_SAD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Average Difference feature.
    """
    return np.abs(ic.select_weights(FA).mean()) - np.abs(ic.select_weights(PA).mean())


def compute_SVD(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Variance Difference feature.
    """
    return ic.select_weights(FA).var() - ic.select_weights(PA).var()


def compute_SED(ic) -> float:
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Spatial Eye Difference feature.
    """
    return np.abs(ic.select_weights(LE).mean() - ic.select_weights(RE).mean())


def compute_MIF(ic):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Myogenic identification feature.
    """
    freqs, psd = ic.psd(verbose=False)
    mean_psd = psd.mean(axis=0)
    return mean_psd[freqs > 20].sum() / mean_psd.sum()


def compute_CORR_BL(ic, thres=0.65):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Correaltion with eye blink example
    """
    in_epoch_corrs = [_cross_corr(eye_blink_example, epoch) for epoch in ic.signal.values]
    pattern_probs = [np.mean(abs(epoch_corr) > thres) for epoch_corr in in_epoch_corrs]

    return np.mean(pattern_probs)


def compute_CORR_MOVE(ic, thres=0.65):
    """
    Args:
        ic (IC): indepentent component.

    Returns:
        float: Correaltion with eye movement example
    """
    in_epoch_corrs = [_cross_corr(eye_move_example, epoch) for epoch in ic.signal.values]
    pattern_probs = [np.mean(abs(epoch_corr) > thres) for epoch_corr in in_epoch_corrs]

    return np.mean(pattern_probs)


def compute_CIF(ic):
    # TODO Implement feature. Address low frequency resolution
    raise NotImplementedError


default_features = {'K': compute_K,
                    'MEV': compute_MEV,
                    'SAD': compute_SAD,
                    'SVD': compute_SVD,
                    'SED': compute_SED,
                    'MIF': compute_MIF,
                    'CORR_BL': compute_CORR_BL,
                    'CORR_MOVE': compute_CORR_MOVE,
                    'AT':compute_AT,
                    'MT':compute_MT,
                    'AMALB':compute_AMALB}


def build_feature_df(data, default=True, custom_features={}):
    """
    Computes the feature matrix for the dataset of components.

    Args:
        data (dataset): A mapping of {ic_id: IC}. Compatible with the dataset representaion produced by load_dataset().
        default (bool, optional): Determines wether to compute a standard selection of features for the dataset. Defaults to True.
        custom_features (dict, optional): A mapping of custom features that will be computed for the dataset.
        The format is {feature_name: compute_feature} where compute_feature is a function with the only argument IC. Defaults to {}.

    Returns:
        pd.Dataframe: The feature matrix for the dataset.
    """
    feature_df = pd.DataFrame(index=data.keys())
    if default:
        for feature_name, compute_feature in default_features.items():
            feature_df[feature_name] = [compute_feature(ic) for ic in data.values()]

    for feature_name, compute_feature in custom_features.items():
        feature_df[feature_name] = [compute_feature(ic) for ic in data.values()]
    return feature_df