#-*- coding: utf-8 -*-
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

import pandas as pd

# 419样本
# df_classes_is0123 = pd.read_csv("./input_csv/df_list_count_right.csv",sep=',',names =None,header=None)
# df_classes_is0123 = pd.DataFrame()
# 622样本，带有杂样本
df_classes_is0123 = pd.read_csv("./input_csv/df_list_all_911.csv",sep=',',names =None,header=None)

x_df = df_classes_is0123.loc[:, 1:2]
y_df = df_classes_is0123.loc[:, 10].astype(np.float64)
y_pre_df = df_classes_is0123.loc[420:, 10].astype(np.float64)

min_x0 = df_classes_is0123.loc[:,0].min()
max_x0 = df_classes_is0123.loc[:,0].max()
min_x1 = df_classes_is0123.loc[:,1].min()
max_x1 = df_classes_is0123.loc[:,1].max()
print(x_df)

import os

errors_numbers = list(range(622))

xlabel_1 = {r"半径",r"面积",r""}

for i in range(9):
    for j in range(i+1,10):
        plt.cla()
        plt.figure(1)

        str_title = "correlation between " + str(i) + "_" + str(j)
        plt.title(str_title)
        plt.xlabel()
        # plt.scatter(df_classes_is0123.loc[:,0],df_classes_is0123.loc[:,1])
        # if (df_classes_is0123.loc[:,10]==df_classes_is0123.loc[:,11]):
        # plt.scatter(df_classes_is0123.loc[y_df==0,i],df_classes_is0123.loc[y_df==0,j],color="r",alpha=0.3)
        # plt.scatter(df_classes_is0123.loc[y_df==1,i],df_classes_is0123.loc[y_df==1,j],color="g",alpha=0.3)
        # plt.scatter(df_classes_is0123.loc[y_df==2,i],df_classes_is0123.loc[y_df==2,j],color="b",alpha=0.3)
        # plt.scatter(df_classes_is0123.loc[y_df==3,i],df_classes_is0123.loc[y_df==3,j],color="c",alpha=0.3)
        for ii in errors_numbers:

            # if (df_classes_is0123.loc[ii,10]==df_classes_is0123.loc[ii,11] and df_classes_is0123.loc[ii,10]==0):
            #     plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="r",alpha=0.3)
            # if (df_classes_is0123.loc[ii,10]==df_classes_is0123.loc[ii,11] and df_classes_is0123.loc[ii,10]==1):
            #     plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="g",alpha=0.3)
            # if (df_classes_is0123.loc[ii,10]==df_classes_is0123.loc[ii,11] and df_classes_is0123.loc[ii,10]==2):
                # plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="b",alpha=0.3)
            if (df_classes_is0123.loc[ii,10]==df_classes_is0123.loc[ii,11] and df_classes_is0123.loc[ii,10]==3):
                plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="c",alpha=0.3)

            # elif (df_classes_is0123.loc[ii,10]!=df_classes_is0123.loc[ii,11] and df_classes_is0123.loc[ii,10]==0):
            #     plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="r",marker="+",alpha=0.5)
            # elif (df_classes_is0123.loc[ii, 10] != df_classes_is0123.loc[ii, 11] and df_classes_is0123.loc[ii, 10] == 1):
            #     plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="g",marker="+",alpha=0.5)
            # elif (df_classes_is0123.loc[ii, 10] != df_classes_is0123.loc[ii, 11] and df_classes_is0123.loc[ii, 10] == 2):
            #     plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="b",marker="+",alpha=0.5)
            elif (df_classes_is0123.loc[ii, 10] != df_classes_is0123.loc[ii, 11] and df_classes_is0123.loc[ii, 10] == 3):
                plt.scatter(df_classes_is0123.loc[ii, i], df_classes_is0123.loc[ii, j], color="c",marker="+",alpha=0.5)

        # plt.scatter(df_classes_is0123.loc[y_pre_df==0, i], df_classes_is0123.loc[y_pre_df==0, j], color="r",marker="+")
        # plt.scatter(df_classes_is0123.loc[y_pre_df==1, i], df_classes_is0123.loc[y_pre_df==1, j], color="g",marker="+")
        # plt.scatter(df_classes_is0123.loc[y_pre_df==2, i], df_classes_is0123.loc[y_pre_df==2, j], color="b",marker="+")
        # plt.scatter(df_classes_is0123.loc[y_pre_df==3, i], df_classes_is0123.loc[y_pre_df==3, j], color="c",marker="+")
            # print(i)
        str_base = "output_statisticImage/plot_scatter_contrast_classes_3_"
        if not os.path.exists(str_base):
            os.mkdir(str_base)
        # str1 = "output_statisticImage/plotscatter_row_" + str(i) + "_col_" + str(j) + ".jpg"
        str1 = "output_statisticImage/plot_scatter_contrast_classes_3_/plotscatter_row_" + str(i) + "_col_" + str(j) + "with_error.png"
        plt.savefig(str1,dpi=300)
        # plt.show()
