import pandas as pd
import warnings
from pandas import DataFrame
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.dates as mdates
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from parser_utils import folder_selection_utils
from scipy.stats import linregress
import os
import math
import parser_utils.parser_logger
import logging
import json
# TODO split DFs if time gap > 60s? and plot separately (assume if car not move > 60s, it's separate runs)
# TODO shade eff regions?
# TODO calc time spent in torque/rpm ranges and get average efficiency
# TODO write a script that will generate curves of different power limting methods
    # compare torque limit, power limit, current limit
    # generate torque vs rpm for each
warnings.simplefilter(action='ignore',category=pd.errors.PerformanceWarning)
def lpf_df(df:DataFrame,cutoffhz=10):
    # Define the filter
    order = 2
    fs = 1 / (10 / 1000)  # Sampling frequency
    nyquist = fs / 2
    cutoff = min(0.5 * nyquist, cutoffhz)  # Desired cutoff frequency of the filter (Hz)
    print(str(cutoff)+"Hz")
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    for col in list(df):
        df["filt"+col] = filtfilt(b, a, df[col])
        # plt.plot(df.index,df["filt"+col],label="filt")
        # plt.plot(df.index,df[col],label="normal")
        # plt.legend()
        # plt.show()
    return df


def drop_rows_from_df(df: DataFrame, column_name, min, max):
    newdf = df[(df[column_name] >= min) & (df[column_name] <= max)]
    rows_dropped = len(df) - len(newdf)
    logging.info(f"{column_name}: dropped {rows_dropped} rows")
    return newdf


def resample_data(df: DataFrame, time_column_name = None,resample_interval=10,plots_to_compare=False):
    df = df
    # Assuming 'df' is your original DataFrame with 'Time' as epoch time in milliseconds
    if time_column_name is not None:
        df[time_column_name] = pd.to_datetime(
            df[time_column_name], unit='ms')  # Convert epoch time to datetime
        df.set_index(time_column_name, inplace=True)  # Set 'Time' as index
    df = df.interpolate()
    # df.to_csv('interpdata.csv')
    # Assuming 'D3_VAB_Vd_Voltage' is the column containing the signal

    df = df[~df.index.duplicated()]
    # Resample to 100ms intervals and forward fill missing values
    interval = str(resample_interval)+"L"
    df_resampled = df.resample(interval, convention='start').ffill()

    # Save resampled DataFrame to a file
    # df_resampled.to_csv('resampled_data.csv')
    if plots_to_compare:
        for series in list(df_resampled):
            if series in list(df):
                plt.plot(df.index, df[series], label='Original')
                plt.title(series + ' vs Time (Original)')
                plt.xlabel('Time')
                plt.ylabel(series)
                plt.legend()
                # # Plot resampled data
                plt.plot(df_resampled.index,
                        df_resampled[series], label='Resampled',alpha=0.8)
                plt.title(series+' vs Time (Resampled)')
                plt.xlabel('Time')
                plt.ylabel(series)
                plt.legend()
                plt.tight_layout()
                plt.show()
            else:
                logging.error(f"{series} not found in {list(df)}")
    # Show plots

    return df_resampled


def sort_df_closed_shape(dfe: DataFrame):
    # Compute centroid
    cent = (dfe['X'].mean(), dfe['Y'].mean())

    # Sort by polar angle
    dfe['angle'] = dfe.apply(lambda row: math.atan2(
        row['Y'] - cent[1], row['X'] - cent[0]), axis=1)
    df_sorted = dfe.sort_values(by='angle')
    return df_sorted


def get_eff_curve_patches(dfs: "list[DataFrame]", names: "list[str]", ax=plt.gca()):
    def computeArea(polygon:patches.Polygon):
        # https://stackoverflow.com/questions/62427366/computing-the-area-filled-by-matplotlib-pyplot-fill
        pos=polygon.xy
        x, y = (zip(*pos))
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    patch_list = {}
    for i in range(len(dfs)):
        curve = dfs[i]
        curvename = names[i]
        if "red" in curvename:
            color = "crimson"
        elif "purple" in curvename:
            color = "mediumpurple"
        elif "green" in curvename:
            color = "chartreuse"
        elif "inner" in curvename:
            color = "lightskyblue"
        elif "outer" in curvename:
            color = "cornflowerblue"
        curve = sort_df_closed_shape(curve)
        patch = patches.Polygon(
            curve[['X', 'Y']].values, closed=True, fill=False, edgecolor=color,alpha=1.0,ls="-",lw=1)
        # ax.add_patch(patch)
        # ax.scatter(curve["X"], curve["Y"], label=curvename,
        #            c=color, marker='.', alpha=0.1)
        area=computeArea(patch)
        patch_list[curvename] = (patch,area)
    patch_list= dict(sorted(patch_list.items(), key=lambda item: item[1][1],reverse=True))
    return patch_list


def annot_max(df: DataFrame, seriesname, ax=None):
    ymax = df[seriesname].max()
    xmax = df[seriesname].idxmax()
    if type(xmax) != np.int64:
        xmax = xmax.to_numpy()

    text = "max={:.2f}v".format(ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="simple", facecolor='black')
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, ha="center", va="top")
    ax.annotate(text, xy=(mdates.date2num(xmax), ymax),
                xytext=(0.15, -0.2), **kw)


def annot_min(df: DataFrame, seriesname, ax=None):
    ymin = df[seriesname].min()
    xmin = df[seriesname].idxmin()
    xmin = xmin.to_numpy()

    text = "min={:.3f}v".format(ymin)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="simple", facecolor='black')
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops,  ha="center", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(.85, -0.2), **kw)


def main(pick_folder=True, crawl_paths=False):
    
    parser_utils.parser_logger.setup_logger(verbose=False)
    plt.style.use('bmh')
    real_list = set()

    if pick_folder:
        folder = folder_selection_utils.select_folder_and_get_path()
        real_list.add(folder)
    elif crawl_paths:
        # Specify the path to the exported JSON file
        json_file_path = r'C:\Users\Matthew Samson\source\repos\KS5e-Data-Logging\6eLogsfiles.json'

        # Load the list of paths from the JSON file
        with open(json_file_path, 'r') as json_file:
            folder_list = json.load(json_file)

        for file in folder_list:
            logging.debug(file)
            real_list.add(os.path.dirname(file))
    else:
        logging.info("no option selected")
        return

    eff_curve_dfs = []
    eff_curve_names = []
    for eff_curve in os.listdir("efficiency_plot"):
        if eff_curve.endswith('.csv') or eff_curve.endswith('.CSV'):
            curve_path = os.path.join("efficiency_plot", eff_curve)
            curve_df = pd.read_csv(curve_path)
            eff_curve_dfs.append(curve_df)
            eff_curve_names.append(str(eff_curve))
            logging.debug(curve_df)
            
    df_list = []
    for folder in real_list:
        outfolder = r'D:/MatthewS/motor_analysis_fsae/round2/6e/'
        outfolder_set = False
        for filename in os.listdir(folder):
            logging.info(f"Folder: {folder}")
            if filename.endswith('.csv') or filename.endswith('.CSV'):
                file_path = os.path.join(folder, filename)
                try:
                    df = pd.read_csv(file_path)
                    # df = df[['D2_Motor_Speed','D2_Torque_Feedback','Time','D1_VSM_State','D2_Motor_Speed','D2_Torque_Feedback','D1_Commanded_Torque','D4_Iq','D3_Id','D1_DC_Bus_Voltage','D4_DC_Bus_Current']]
                    if ("D2_Torque_Feedback") not in list(df):
                        logging.info(f"Skipping {file_path} because no torque feedback")
                        continue
                    # if df["D2_Motor_Speed"].max() < 1000:
                    #     logging.info(f"Skipping {file_path} because motor speed less than 1000 max")
                    #     continue
                    # if df["D2_Torque_Feedback"].max() < 200:
                    #     logging.info(f"Skipping {file_path} because torque feedback less than 200")
                    #     continue
                    # if df["D1_Commanded_Torque"].max() < 200:
                    #     logging.info(f"Skipping {file_path} because torque command max less than 200")
                    #     continue
                    try:
                        df = resample_data(df, "Time")
                    except KeyError as e:
                        logging.error(f"Key missing {e} in {filename}")
                        continue
                    try:
                        df = drop_rows_from_df(df, "D1_VSM_State", 4, 7)
                    except KeyError as e:
                        logging.error(f"Key missing {e} in {filename}")
                        continue
                    try:
                        df = drop_rows_from_df(
                            df, "D2_Motor_Speed", -10, 7000)
                    except KeyError as e:
                        logging.error(f"Key missing {e} in {filename}")
                        continue
                    df = lpf_df(df,100)
                    realistic_torque_max = df['D1_Commanded_Torque'].max()
                    try:
                        df = drop_rows_from_df(
                            df, "D2_Torque_Feedback", 0, realistic_torque_max)
                    except KeyError as e:
                        logging.error(f"Key missing {e} in {filename}")
                        continue


                    try:
                        df = drop_rows_from_df(df, "Torque_Command", 10, 400)
                    except:
                        logging.error(
                            f"using torque command (0xc0) to filter failed {filename}")
                        try:
                            df = drop_rows_from_df(
                                df, "D1_Commanded_Torque", 10, 400)
                        except:
                            logging.error("damn LMAO!")
                    
                    # if DF is super teeny, skip it 
                    logging.info(len(df))
                    if len(df) <= 10:
                        logging.info("fuck this df")
                        continue
                    # Attemp to split df into "runs" if over 45s passed and no points
                    runs_dfs = []
                    df['groups'] = (
                        df.index.to_series().diff().dt.seconds > 45).cumsum()
                    
                    for ct, data in df.groupby('groups'):
                        runs_dfs.append(data)

                    for index,run_data in enumerate(runs_dfs):
                        earliest_timestamp = run_data.index.min()
                        # if run_data["D2_Motor_Speed"].max() < 2000:
                        #     continue
                        # if run_data["D1_Commanded_Torque"].max() < 200:
                        #     continue
                        # Set up figure
                        fig = plt.figure(figsize=(16.2, 9.1))
                        ax1 = fig.add_subplot(4, 4, (1, 12))
                        ax2 = fig.add_subplot(4, 4, (13, 14))
                        ax3 = fig.add_subplot(4, 4, (15, 16))
                        

                        ax1.text(3270, 95, "96%+", fontsize=12)
                        ax1.text(2800, 155, "95%+", fontsize=12)
                        ax1.text(3200, 190, "94%+", fontsize=12)
                        ax1.text(3200, 220, "90-94%+", fontsize=12)
                        ax1.text(2400, 23, "86-90%+", fontsize=12)
                        ax1.scatter(run_data['filtD2_Motor_Speed'], run_data['filtD1_Commanded_Torque'],
                                    marker='o', label="Command Torque", c='blue', alpha=0.9)
                        ax1.scatter(run_data['filtD2_Motor_Speed'], run_data['filtD2_Torque_Feedback'],
                                    marker='o', label="Feedback Torque", c='aqua', alpha=0.9)

                        ax1.set_xlabel("RPM")
                        ax1.set_ylabel("Torque Command and Est. Torque (Nm)")
                        ax1.set_xlim([0, 7000])
                        ax1.set_ylim([0, 250])

                        ax1_1 = ax1.twinx()
                        ax1_1.scatter(run_data["D2_Motor_Speed"], run_data["D4_Iq"],
                                    label="Iq", marker='.', c='tomato', alpha=0.1)
                        ax1_1.scatter(run_data["D2_Motor_Speed"], abs(run_data["D3_Id"]),
                                    label="Id", marker='.', c='yellowgreen', alpha=0.1)
                        ax1_1.set_ylabel("Q and D axis current (Amps)")

                        nticks = 10
                        ax1.yaxis.set_major_locator(
                            matplotlib.ticker.LinearLocator(nticks))
                        ax1_1.yaxis.set_major_locator(
                            matplotlib.ticker.LinearLocator(nticks))

                        iq_rms = run_data["D4_Iq"] / (math.sqrt(2))

                        ax2.scatter(
                            iq_rms, run_data["D2_Torque_Feedback"], label="Torque")

                        slope, intercept, r_value, p_value, std_err = linregress(
                            iq_rms, run_data['D2_Torque_Feedback'])
                        line = slope * iq_rms + intercept
                        ax2.plot(iq_rms, line, color='tab:red', linestyle='--',
                                label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f} (r:{r_value} p: {p_value} err: {std_err})')

                        ax2.set_xlabel('Q-axis Current (RMS)')
                        ax2.set_ylabel('Torque Feedback')
                        # ax2.tick_params(axis='y', labelcolor='tab:green')

                        ax3.scatter(run_data.index, run_data['D1_DC_Bus_Voltage'],
                                    label='DC Bus Voltage (V)', marker='.')
                        ax3.scatter(run_data.index, run_data["D4_DC_Bus_Current"],
                                    label="DC Bus Current (A)", marker='.')
                        ax3.scatter(
                            run_data.index, run_data["D2_Motor_Speed"]/10, label="RPM/10", marker='.')
                        
                        # Plot eff. regions last so they are on the top
                        patches = get_eff_curve_patches(eff_curve_dfs,eff_curve_names,ax1)
                        for i in patches:
                            logging.debug(patches[i][0])
                            ax1.add_patch(patches[i][0])
                            
                        annot_max(run_data, "D1_DC_Bus_Voltage", ax=ax3)
                        annot_min(run_data, "D1_DC_Bus_Voltage", ax=ax3)

                        ax3.set_xlabel('Time')
                        ax3.set_ylabel('Volts, Amps, RPM')

                        ax1.legend(loc='upper left')
                        ax1_1.legend(loc='upper right')
                        ax2.legend(loc='upper left')
                        ax3.legend(loc='lower center')

                        title_str = "Torque, Current, RPM of log: "+filename+" timestamp: " + str(earliest_timestamp)
                        title_str += " run: "+str(index) if len(runs_dfs) > 1 else ""
                        plt.title(title_str)
                        plt.tight_layout()
                        file_friendly_timestamp = str(
                            earliest_timestamp.strftime("%Y_%m_%d-%H-%M-%S"))
                        file_friendly_filename = filename.replace(".", "")

                        if not outfolder_set:
                            outfolder += str(
                                earliest_timestamp.strftime("%Y_%m"))
                            try:
                                os.makedirs(outfolder, exist_ok=True)
                            except:
                                logging.debug("lmao")
                            outfolder_set = True
                        export_filename = f"kt{slope:.2f}_{file_friendly_timestamp}_{file_friendly_filename}"
                        export_filename += "_run_"+str(index) if len(runs_dfs) > 1 else ""
                        plt.close()
                        plt.plot(run_data.index,run_data['D2_Motor_Speed'],marker='.')
                        plt.plot(run_data.index,run_data['D1_Commanded_Torque'],marker='.')
                        plt.title(title_str)
                        # plt.show()
                        run_data.to_csv(f"{file_friendly_filename}_run_{(index)}.csv")
                        # plt.savefig(os.path.join(outfolder, export_filename+".png"))
                        # plt.close()
                        # run_data.to_csv(os.path.join(
                        #     outfolder, export_filename+".csv"), sep=",")
                        # mng = plt.get_current_fig_manager()
                        # mng.full_screen_toggle()
                        # plt.show()
                        # run_data_list.append(run_data)

                except (ValueError,KeyError) as e:
                    logging.error(f"error with {file_path}, {e}")
        # big_df = pd.concat(df_list)
        # fig, (ax1, ax2) = plt.subplots(2, figsize=(16.2, 9.1))
        # ax1.scatter(big_df['D2_Motor_Speed'], big_df['D1_Commanded_Torque'],
        #             marker='o', label="Command Torque", c='blue')
        # ax1.scatter(big_df['D2_Motor_Speed'], big_df['D2_Torque_Feedback'],
        #             marker='o', label="Feedback Torque", c='royalblue')
        # ax1.set_xlabel("RPM")
        # ax1.set_ylabel("Torque Command and Est. Torque (Nm)")
        # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax1_1 = ax1.twinx()
    # ax1_1.scatter(big_df["D2_Motor_Speed"], big_df["D4_Iq_Command"],
    #               label="Iq Command", marker='.', c='maroon')
    # ax1_1.scatter(big_df["D2_Motor_Speed"], abs(big_df["D3_Id_Command"]),
    #               label="Id Command", marker='.', c='darkolivegreen')
    # ax1_1.scatter(big_df["D2_Motor_Speed"], abs(
    #     big_df["D2_Flux_Weakening_Output"]), label="Flux Weakening Output", c='palegreen', marker='.')
    # ax1_1.scatter(big_df["D2_Motor_Speed"], big_df["D4_Iq"],
    #               label="Iq", marker='.', c='tomato')
    # ax1_1.scatter(big_df["D2_Motor_Speed"], abs(big_df["D3_Id"]),
    #               label="Id", marker='.', c='yellowgreen')
    # ax1_1.set_ylabel("Q and D axis current (Amps)")

    # nticks = 10
    # ax1.yaxis.set_major_locator(
    #     matplotlib.ticker.LinearLocator(nticks))
    # ax1_1.yaxis.set_major_locator(
    #     matplotlib.ticker.LinearLocator(nticks))
    # # ax1_1.tick_params(axis='y', labelcolor='tab:red')

    # iq_rms = big_df["D4_Iq"] / (math.sqrt(2))

    # ax2.scatter(iq_rms, big_df["D2_Torque_Feedback"], label="Torque")

    # slope, intercept, r_value, p_value, std_err = linregress(
    #     iq_rms, big_df['D2_Torque_Feedback'])
    # line = slope * iq_rms + intercept
    # ax2.plot(iq_rms, line, color='tab:red', linestyle='--',
    #          label=f'Linear Fit: y = {slope:.2f}x + {intercept:.2f}')

    # ax2.set_xlabel('Q-axis Current (RMS)')
    # ax2.set_ylabel('Torque Feedback')
    # # ax2.tick_params(axis='y', labelcolor='tab:green')

    # ax1.legend(loc='upper left')
    # ax1_1.legend(loc='upper right')
    # ax2.legend(loc='upper left')
    # plt.title("Torque, Current, RPM of all logs, timestamp: " +
    #           str(earliest_timestamp))
    # plt.tight_layout()
    # file_friendly_timestamp = str(
    #     earliest_timestamp.strftime("%Y_%m_%d-%H-%M-%S"))
    # file_friendly_filename = filename.replace(".", "")
    # export_filename = f"kt{slope:.2f}_{file_friendly_timestamp}"
    # plt.savefig(os.path.join(folder, export_filename+".png"),
    #             bbox_inches='tight')
    # big_df.to_csv(outfolder+export_filename+".csv",sep=",")
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()


if __name__ == "__main__":
    main(pick_folder=True,crawl_paths=False)
