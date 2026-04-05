import get_embedding

config_path = "configs/exp_029_finetune.yaml"
weights_path = "../results/exp_029/exp_029_finetune.pt"

# Whether to only compute embeddings for the labeled subset of the data.
labeled_only = True

# A list of filepaths to load. If None, will load all npz files in data_dir.
#filepaths = glob.glob('../data/*5.npz')    # 3/15/26, ran code limiting to files ending in 5.npz. probably a better way to randomly select
filepaths = None

get_embedding.get_embedding(
    config_path=config_path,
    weights_path=weights_path,
    labeled_only=labeled_only,
    filepaths=filepaths,
    data_dir="../data",
    output_dir="../data"
    )
