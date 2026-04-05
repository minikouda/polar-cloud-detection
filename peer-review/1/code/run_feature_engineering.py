import feature_engineering

data_dir = "../data"
path_output = "../data"
embedding_dir = "../data"

# Whether to only compute embeddings for the labeled subset of the data.
labeled_only = True

# A list of filepaths to load. If None, will load all npz files in data_dir.
filepaths = []

feature_engineering.feature_engineering(
    data_dir=data_dir,
    path_output=path_output,
    embedding_dir=embedding_dir,
    labeled_only=labeled_only,
    filepaths=filepaths,
)
