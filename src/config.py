# CSV files directory
DATA_PATH = "..\\data"

#seed
seed = 0


lgb_params_grid = {
    "objective" : ["multiclass"],
    "metric": ['multi_logloss'],
    "num_class" :[12],
    "max_depth": [64,128],
    "num_leaves" : [255,400],
    "bagging_fraction" : [0.9],
    "feature_fraction" : [0.9],
    "bagging_freq" : [5,10],
    "max_bin" : [250,400],
    "learning_rate" : [0.05,0.1],
    "bagging_seed" : [seed],
    "verbosity" : [-1]
}
