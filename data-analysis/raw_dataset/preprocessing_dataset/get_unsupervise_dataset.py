import pandas as pd 
from get_test_train_dataset import text_normalize
from dotenv import load_dotenv
import os 
load_dotenv('../.env')

pretrain_df = pd.read_csv(os.getenv('RAW_UNSUPERVISED_DATASET_PATH'), encoding='latin-1', header=None)[5].apply(text_normalize)
pretrain_df.to_csv(os.getenv('CLEANED_UNSUPERVISED_DATASET_PATH'), index=False, header=False)

