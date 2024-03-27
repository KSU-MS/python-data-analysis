import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import sys
# sys.path.append('parser_utils')
from parser_utils.folder_selection_utils import *
import matplotlib.pyplot as plt 
import numpy as np
import numpy.ma as ma
import os

# parse "betterParsed..." for only cell temps and few other specified signals (the ones in new_headers)
df = pd.read_csv(select_file_and_get_path())
# OPTIONAL truncate for a certain period - these args are indices of the dataframe.
df = df.truncate(before=0,after=(len(df)))
headers = list(df)
new_headers=["Time","Pack_Summed_Voltage","Pack_Current","D2_Motor_Speed",
            "Low_Temperature","High_Temperature"]
for i in headers:
    if ('temp' in i) and ('cell' in i):
        new_headers.append(i)
new_headers.sort()
new_df = df[new_headers]
# new_df=pd.DataFrame(columns=new_headers)
# for i in new_headers:
#     new_df[i]=df[i]
print(new_df)
# Output file of just signals we want, but with raw cell temp sensor voltages
# newfile="temps_raw.csv"
# with open(newfile,"w") as f:
#     new_df.to_csv(newfile)

# new_df=pd.read_csv("temps_raw.csv")
new_headers=list(new_df)
print(new_headers)

def polynomial_fit_temp(temp):
    for i in range(len(temp)):
        if temp[i] > 0:
            if temp[i] > 3:
                temp[i]=temp[i]/0.019607 # Convert from bits to volts
            temp[i] = round((-0.000002416676401)*temp[i]**5+0.001082617446913*temp[i]**4+(-0.194488265848684)*temp[i]**3+(17.5197709028014)*temp[i]**2+(-792.865188960333)*temp[i]+14494.8611005946,4)
        else:
            continue
    return temp

for i in range(7,len(new_headers)):
    new_df[new_headers[i]]=polynomial_fit_temp(new_df[new_headers[i]])

# newfile2="temps_real_newcal.csv"
# with open(newfile2,"w") as f: 
#     new_df.to_csv(newfile2)

# elif os.path.isfile("temps_real_newcal.csv"):

# new_df=pd.read_csv("temps_real_newcal.csv")
new_df=new_df.interpolate()
new_headers=list(new_df)

newfile3="temps_real_newcal_interpolated.csv"
# with open(newfile3,"w") as f: 
#     new_df.to_csv(newfile3)

print(new_headers)

fig, (ax1) = plt.subplots(1,sharex=True)

new_df["Time"]/=1000

def plot_vs_time(plot,thing,ls='-'):
    try:
        plot.plot(new_df['Time'],thing,ls,label=thing.name)
        print("Plotted: " + thing.name)
    except: 
        plot.plot(new_df['Time'],thing,ls)
    
for i in new_headers:
    if ('temp' in i) and ('cell' in i):
        plot_vs_time(ax1,new_df[i])

ax1.set_ylim((0,70))
# for i in range(1,6):
#     plot_vs_time(ax2,new_df[new_headers[i]])


# plt.legend()
plt.show()
# else:
#     print("how did you even get here")