

def clean(df, test=0):
    """
    Clean and preprocess MISR cloud classification data.

    Remove unlabeled observations (label == 0)
    Keep only the most informative features: NDAI, SD, CORR
    Standardize features

    df : pd.DataFrame
        ['x', 'y', 'ndai', 'sd', 'corr',
         'radiance_df', 'radiance_cf', 'radiance_bf',
         'radiance_af', 'radiance_an', 'label', 'image_data']

    """

    df = df.copy()

    # remove unlabeled data for training data
    if not test: 
        df = df[df["label"] != 0]

    # normalize data
    features_to_normalize = ['sd', 'radiance_af', 'radiance_an', 
                         'radiance_bf', 'radiance_cf', 
                         'radiance_df']

    for feat in features_to_normalize: 
        normalize(df, feat)

   
    return df


def normalize(df, feature): 
    #df = df.copy()
    df_feature_mean = df[feature].mean()
    df_feature_sd = df[feature].std()

    df[f'normalized_{feature}'] = (df[feature] - df_feature_mean) / df_feature_sd