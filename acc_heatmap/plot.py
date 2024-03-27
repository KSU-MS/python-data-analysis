import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import numpy.ma as ma
df = pd.read_csv(r"cardata/ks6e/acc-test-no-fans-data/temps_raw.csv")
df["Time"]/=1000

fig, (ax1,ax2) = plt.subplots(2,sharex=True)
ax1_1 = ax1.twinx()

def plot_vs_time(plot,thing,ls='-'):
    try:
        plot.plot(df['Time'],thing.interpolate(),ls,label=thing.name)
    except: 
        plot.plot(df['Time'],thing,ls)

plot_vs_time(ax1_1,df["D6_Inverter_Enable_State"])
plot_vs_time(ax1_1,df["D7_BMS_Torque_Limiting"])
ax1_1.set_ylim([0,10])

plot_vs_time(ax1,df["D1_DC_Bus_Voltage"])
plot_vs_time(ax1,df["Pack_Summed_Voltage"])
plot_vs_time(ax1,df["Pack_Current"])
plot_vs_time(ax1,df["D2_Motor_Speed"]/10)
ax1.axhline(y = 280, color = 'r', linestyle = '-')

plot_vs_time(ax2,df["dash_button1status"])
plot_vs_time(ax2,df["dash_button2status"])
plot_vs_time(ax2,df["dash_button3status"])

df["real_pwr"]= df["Pack_Current"]*df["Pack_Summed_Voltage"]
df["command_pwr"] = df["D1_Max_Discharge_Current"]*df["D1_DC_Bus_Voltage"]
ax1.set_xlabel("time")
ax2.set_xlabel("time")
fig.tight_layout()

plt.grid(True,which="both",axis="both",figure=fig)

fig2, ax3 = plt.subplots(1)

df["real_pwr_filter"]=df["real_pwr"].dropna().rolling(10).mean()
mask = ma.masked_less(df["real_pwr"].dropna().to_numpy(),17001)
# print(mask)
plot_vs_time(ax3,df["real_pwr_filter"],ls='.')
plot_vs_time(ax3,df["real_pwr"])
# ax3.plot(df["Time"],mask,'y')
plot_vs_time(ax3,df["command_pwr"])
ax3.axhline(y = 17000, linestyle='dashed')


ax3.set_xlabel("time")
plt.legend()
plt.show()