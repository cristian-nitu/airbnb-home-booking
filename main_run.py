"""
the main entrance of the project
"""

from sklearn.model_selection import KFold
from models import *
import config
import pickle
import gc
from preprocessing import load_data
#from feature_engineering import extract_features
#from utils import reduce_mem_usage

