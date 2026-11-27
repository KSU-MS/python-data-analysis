import platform
import os
import subprocess
import tkinter as tk
from tkinter import filedialog
import logging
import shutil

def select_folder_and_get_path():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory()
    
    if folder_path:
        logging.info(f"Selected folder path: {folder_path}")
        return folder_path
    else:
        logging.warning("No folder selected")
        return None
    
def select_file_and_get_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfile()
    
    if file_path:
        print(f"Selected file path: {file_path}")
        return file_path
    else:
        print("No file selected")
        return None
    
def select_folder_and_get_path_dbc():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes("-topmost", True)
    folder_path = filedialog.askdirectory()
    
    if folder_path:
        logging.info(f"Selected folder path: {folder_path}")
        return folder_path
    else:
        logging.warning("No folder selected")
        return None
    
def open_path(path):
    system = platform.system()
    if system == 'Windows':
        logging.debug("detected that operating system is Windows")
        path = '"'+path+'"'
        os.startfile(path)
        # subprocess.run(["cmd.exe","start", path])
    elif system == 'Linux':
        logging.debug("detected that operating system is Linux")
        subprocess.run(["xdg-open", f"{path}"])
    elif system == 'Darwin':
        logging.debug("detected that operating system is MacOS")
        # os.system(f"xdg-open {path}")  
    else:
        logging.error("Unknown operating system")
    
def copy_file_with_prefix(src, dest, prefix):
    # Copy the file
    logging.info(os.getcwd())
    shutil.copy(src, dest)

    # Get the filename from the source path
    filename = os.path.basename(src)

    # Construct the new filename with prefix
    new_filename = prefix + filename

    # Get the full path of the copied file
    copied_file_path = os.path.join(dest, filename)

    # Rename the copied file with the new filename
    os.rename(copied_file_path, os.path.join(dest, new_filename))

# # Call the method to select a folder and get its path
# selected_folder_path = select_folder_and_get_path()

# if selected_folder_path:
#     # Use the selected folder path for further operations
#     pass
