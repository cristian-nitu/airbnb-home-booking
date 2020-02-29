import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import glob
from datetime import datetime

# TODO - Data Exploration
from glob import glob
import os
import pandas as pd

def load_data(directory):
   file_names = glob(directory +'*.csv')
   return {os.path.basename(f): pd.read_csv(f) for f in file_names}

data = load_data('\\Users\\cristiann\\desktop\\assigments\\hotel_booking-master\\')



