import pandas as pd
import numpy as np
import pdb
from Bio.Seq import Seq
import sys


def main(f_feature, f_pred):
	df_all= pd.read_csv(f_feature)
	df_info = df_all[['transcript id','guide','target_pos_list','target pos num']]
	df_pred = pd.read_csv(f_pred)
	df_pred_sel = df_pred[['spacer sequence','predicted_value_sigmoid']]
	df_com = (df_info.merge(df_pred_sel, left_on='guide', right_on='spacer sequence')).drop(columns=['spacer sequence'])
	df_com['rank']= df_com.groupby("transcript id")["predicted_value_sigmoid"].rank(ascending=False)
	df_sorted = df_com.sort_values(by=['transcript id','rank'])
	outf = 'results/'+f_feature[:-4].split('/')[-1]+'_prediction_sorted.csv'
	df_sorted.to_csv(outf,index=False)
	#df_sorted.to_csv(f_feature[:-4]+'_guide_prediction_sorted.csv',index=False)


if __name__ == '__main__':
    main(str(sys.argv[1]),str(sys.argv[2]))
