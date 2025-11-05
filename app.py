# """
# Streamlit web application for a 3-step image processing pipeline:
# 1. (Optional) Undistort omnidirectional images from a user-selected input folder.
# 2. Write a config.cfg file based on user inputs from the UI.
# 3. Run an external image stitching executable on the correct set of images.
# """

# import streamlit as st
# import cv2
# import numpy as np
# import os
# from pathlib import Path
# import time
# import sys
# import subprocess

# # Import the OcamModel class
# try:
#     from ocam_model import OcamModel
# except ImportError:
#     st.error("ERROR: Could not import OcamModel from ocam_model.py. Make sure it's in the same directory.")
#     st.stop()

# # --- Core Pipeline Functions (Refactored for Streamlit) ---

# def log_message(log_area, current_log, message):
#     """Helper to append a message to the log display and console."""
#     print(message)  # Log to console for debugging
#     current_log += message + "\n"
#     log_area.code(current_log, language="bash")
#     return current_log

# def empty_directory(directory_path_str, log_area, current_log):
#     """
#     Checks for a directory and deletes all *.jpg files inside it.
#     """
#     dir_path = Path(directory_path_str)
#     current_log = log_message(log_area, current_log, f"Checking output directory: {dir_path}")
#     if dir_path.exists() and dir_path.is_dir():
#         current_log = log_message(log_area, current_log, "Clearing old files from output directory...")
#         files_deleted = 0
#         try:
#             for file_path in dir_path.glob("*.jpg"):
#                 file_path.unlink()  # Delete the file
#                 files_deleted += 1
#             current_log = log_message(log_area, current_log, f"Deleted {files_deleted} old .jpg files.")
#         except Exception as e:
#             current_log = log_message(log_area, current_log, f"Warning: Could not delete all files. {e}")
#     else:
#         current_log = log_message(log_area, current_log, "Output directory not found. It will be created.")
#     return current_log

# def undistort_images(input_dir, output_dir, calib_file, sf_value, log_area, current_log):
#     """
#     Loads calibration, finds/undistorts images, and updates Streamlit UI.
#     """
#     current_log = log_message(log_area, current_log, "--- 1. Starting Image Undistortion ---")
    
#     # --- 1a. Load calibration ---
#     o = OcamModel()
#     current_log = log_message(log_area, current_log, f"Loading calibration from: {calib_file}")
#     if o.load_calib(calib_file) != 0:
#         st.error(f"ERROR: Failed to load calibration file {calib_file}")
#         return False, current_log
    
#     # --- 1b. Create output directory (already cleared, just ensure it exists) ---
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#         current_log = log_message(log_area, current_log, f"Output directory set to: {output_dir}")
#     except OSError as e:
#         st.error(f"ERROR: Could not create directory {output_dir}. {e}")
#         return False, current_log

#     # --- 1c. Get image list ---
#     input_path = Path(input_dir)
#     if not input_path.exists():
#         st.error(f"ERROR: Input directory not found: {input_dir}")
#         return False, current_log
        
#     image_paths = sorted(input_path.glob("*.jpg"))
#     if not image_paths:
#         st.error(f"ERROR: No .jpg images found in directory: {input_dir}")
#         return False, current_log
    
#     current_log = log_message(log_area, current_log, f"Found {len(image_paths)} images. Starting processing...")
    
#     # --- 1d. Loop, undistort, and save ---
#     mapx, mapy = None, None
#     first_image = True
    
#     progress_bar = st.progress(0, "Undistorting images...")
    
#     for i, image_path in enumerate(image_paths):
#         try:
#             src = cv2.imread(str(image_path))
#             if src is None:
#                 current_log = log_message(log_area, current_log, f"Warning: Could not load image: {image_path}")
#                 continue
            
#             if first_image:
#                 height, width = src.shape[:2]
#                 current_log = log_message(log_area, current_log, f"Creating undistortion map (LUT) for size {width}x{height}...")
#                 # Use the sf_value from the UI
#                 mapx, mapy = o.create_perspective_undistortion_lut(height, width, sf_value)
#                 first_image = False

#             dst_persp = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)
#             output_filename = Path(output_dir) / image_path.name
#             cv2.imwrite(str(output_filename), dst_persp)
            
#             percent_complete = (i + 1) / len(image_paths)
#             progress_bar.progress(percent_complete, f"Processing: {image_path.name}")

#         except Exception as e:
#             current_log = log_message(log_area, current_log, f"Error processing {image_path.name}: {e}")
    
#     progress_bar.empty()
#     current_log = log_message(log_area, current_log, "--- Undistortion Complete ---")
#     return True, current_log

# def write_custom_config(config_path, config_values, log_area, current_log):
#     """
#     Creates or overwrites the config.cfg file with values from the Streamlit UI.
#     """
#     current_log = log_message(log_area, current_log, f"--- 2. Writing Custom Config File ({config_path}) ---")
    
#     try:
#         with open(config_path, 'w') as f:
#             for key, value in config_values.items():
#                 f.write(f"{key} {value}\n")
        
#         current_log = log_message(log_area, current_log, "--- Custom Config File Written ---")
#         return True, current_log
#     except Exception as e:
#         st.error(f"ERROR: Could not write config file {config_path}. {e}")
#         return False, current_log

# def run_stitching(stitched_images_dir, executable_path, log_area, current_log):
#     """
#     Finds all .jpg files in the directory and runs the stitching executable.
#     Streams the output in real-time to the Streamlit log area.
#     """
#     current_log = log_message(log_area, current_log, f"--- 3. Starting Image Stitching ---")
    
#     output_path = Path(stitched_images_dir)
#     if not output_path.exists():
#         st.error(f"ERROR: Directory for stitching not found: {stitched_images_dir}")
#         return False, current_log
        
#     images_to_stitch = sorted(output_path.glob("*.jpg"))

#     if not images_to_stitch:
#         st.warning(f"Warning: No .jpg images found in {stitched_images_dir} to stitch.")
#         return True, current_log

#     cmd = [executable_path] + [str(p) for p in images_to_stitch]
#     current_log = log_message(log_area, current_log, f"Running command: {' '.join(cmd[:4])}... ({len(cmd)-1} files total)\n")

#     try:
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1
#         )

#         if process.stdout:
#             for line in iter(process.stdout.readline, ''):
#                 print(line, end='')
#                 current_log += line
#                 log_area.code(current_log, language="bash")

#         process.wait()

#         current_log += f"\n--- Image Stitching Finished (Return Code: {process.returncode}) ---"
#         log_area.code(current_log, language="bash")

#         if process.returncode != 0:
#             st.error(f"ERROR: Image stitching executable failed with code {process.returncode}")
#             return False, current_log
        
#         return True, current_log

#     except FileNotFoundError:
#         st.error(f"ERROR: Executable not found at '{executable_path}'. Make sure it's executable (chmod +x {executable_path}).")
#         return False, current_log
#     except Exception as e:
#         st.error(f"An error occurred while running {executable_path}: {e}")
#         return False, current_log

# # --- Streamlit UI Application ---

# def main_app():
#     st.title("📸 Image Undistortion & Stitching Pipeline")

#     # --- 1. Sidebar for all Configuration ---
#     st.sidebar.title("Pipeline Configuration")

#     # --- Main Settings ---
#     st.sidebar.header("Pipeline Settings")
    
#     input_dir = st.sidebar.text_input(
#         "Input Image Folder", 
#         value="./Set", 
#         help="The folder containing your original .jpg images."
#     )
    
#     st.sidebar.header("Step 1: Undistortion")
#     do_undistort = st.sidebar.toggle("Run Undistortion Step", value=True, help="If disabled, the stitcher will run on the original images from the Input Image Folder.")
#     calibration_type = st.sidebar.radio(
#         "Calibration Type", 
#         ["planar", "fisheye"], 
#         horizontal=True, 
#         disabled=not do_undistort,
#         help="Which calibration file to use. Only active if 'Run Undistortion' is ON."
#     )
    
#     # --- NEW: Added SF_VALUE widget ---
#     sf_value = st.sidebar.number_input(
#         "Zoom Factor (SF_VALUE)",
#         value=4.0,
#         min_value=0.1,
#         step=0.1,
#         format="%.1f",
#         disabled=not do_undistort,
#         help="Zoom factor for perspective undistortion. Only active if 'Run Undistortion' is ON."
#     )
#     # --- End of new widget ---

#     # --- Config File Settings ---
#     st.sidebar.header("Step 2: Stitching Config")
    
#     config_values = {}

#     with st.sidebar.expander("General Modes"):
#         config_values["CYLINDER"] = 1 if st.toggle("CYLINDER", value=False) else 0
#         config_values["ESTIMATE_CAMERA"] = 1 if st.toggle("ESTIMATE_CAMERA", value=False) else 0
#         config_values["TRANS"] = 1 if st.toggle("TRANS", value=True) else 0
#         config_values["ORDERED_INPUT"] = 1 if st.toggle("ORDERED_INPUT", value=True) else 0
#         config_values["CROP"] = 1 if st.toggle("CROP", value=False) else 0
#         config_values["MAX_OUTPUT_SIZE"] = st.number_input("MAX_OUTPUT_SIZE", value=8000, step=100)
#         config_values["LAZY_READ"] = 1 if st.toggle("LAZY_READ", value=True) else 0
#         config_values["FOCAL_LENGTH"] = st.number_input("FOCAL_LENGTH", value=10.0, format="%.2f")

#     with st.sidebar.expander("Keypoint Parameters"):
#         config_values["SIFT_WORKING_SIZE"] = st.number_input("SIFT_WORKING_SIZE", value=1000, step=100)
#         config_values["NUM_OCTAVE"] = st.number_input("NUM_OCTAVE", value=4, step=1)
#         config_values["NUM_SCALE"] = st.number_input("NUM_SCALE", value=7, step=1)
#         config_values["SCALE_FACTOR"] = st.number_input("SCALE_FACTOR", value=1.4142135623, format="%.10f")
#         config_values["GAUSS_SIGMA"] = st.number_input("GAUSS_SIGMA", value=1.4142135623, format="%.10f")
#         config_values["GAUSS_WINDOW_FACTOR"] = st.number_input("GAUSS_WINDOW_FACTOR", value=5, step=1)
#         config_values["CONTRAST_THRES"] = st.number_input("CONTRAST_THRES", value=3e-10, format="%e")
#         config_values["JUDGE_EXTREMA_DIFF_THRES"] = st.number_input("JUDGE_EXTREMA_DIFF_THRES", value=1e-10, format="%e")
#         config_values["EDGE_RATIO"] = st.number_input("EDGE_RATIO", value=100, step=1)
#         config_values["PRE_COLOR_THRES"] = st.number_input("PRE_COLOR_THRES", value=5e-2, format="%e")
#         config_values["CALC_OFFSET_DEPTH"] = st.number_input("CALC_OFFSET_DEPTH", value=4, step=1)
#         config_values["OFFSET_THRES"] = st.number_input("OFFSET_THRES", value=1.0, step=0.1, format="%.1f")

#     with st.sidebar.expander("Descriptor & Matching"):
#         config_values["ORI_RADIUS"] = st.number_input("ORI_RADIUS", value=4.5, format="%.1f")
#         config_values["ORI_HIST_SMOOTH_COUNT"] = st.number_input("ORI_HIST_SMOOTH_COUNT", value=2, step=1)
#         config_values["DESC_HIST_SCALE_FACTOR"] = st.number_input("DESC_HIST_SCALE_FACTOR", value=3, step=1)
#         config_values["DESC_INT_FACTOR"] = st.number_input("DESC_INT_FACTOR", value=512, step=1)
#         config_values["MATCH_REJECT_NEXT_RATIO"] = st.number_input("MATCH_REJECT_NEXT_RATIO", value=0.8, format="%.1f")
#         config_values["RANSAC_ITERATIONS"] = st.number_input("RANSAC_ITERATIONS", value=8000, step=100)
#         config_values["RANSAC_INLIER_THRES"] = st.number_input("RANSAC_INLIER_THRES", value=3.5, format="%.1f")
#         config_values["INLIER_IN_MATCH_RATIO"] = st.number_input("INLIER_IN_MATCH_RATIO", value=0.1, format="%.2f")
#         config_values["INLIER_IN_POINTS_RATIO"] = st.number_input("INLIER_IN_POINTS_RATIO", value=0.04, format="%.2f")
    
#     with st.sidebar.expander("Optimization & Blending"):
#         config_values["STRAIGHTEN"] = 1 if st.toggle("STRAIGHTEN", value=True) else 0
#         config_values["SLOPE_PLAIN"] = st.number_input("SLOPE_PLAIN", value=8e-3, format="%e")
#         config_values["LM_LAMBDA"] = st.number_input("LM_LAMBDA", value=5, step=1)
#         config_values["MULTIPASS_BA"] = st.selectbox("MULTIPASS_BA", [0, 1, 2], index=2)
#         config_values["MULTIBAND"] = st.number_input("MULTIBAND", value=10, step=1)

#     # --- 2. Main Page Layout ---
    
#     log_area = st.code("Waiting to start. Configure settings in the sidebar and click 'Run Pipeline'.", language="bash")
    
#     if st.sidebar.button("🚀 Run Pipeline", type="primary"):
        
#         # --- 3. Run Pipeline Logic ---
#         st.session_state.log_output = "" # Reset log
#         start = time.time()
        
#         # --- Define Paths ---
#         INPUT_DIR = input_dir  # Get from widget
#         OUTPUT_DIR = "./result"
#         CONFIG_FILE = "./config.cfg"
#         STITCHER_EXE = "./image-stitching"
#         # SF_VALUE is now read from the widget
        
#         images_to_stitch_dir = ""

#         # --- Call the empty_directory function ---
#         st.session_state.log_output = empty_directory(OUTPUT_DIR, log_area, st.session_state.log_output)

#         # --- Step 1: (Conditional) Undistortion ---
#         if do_undistort:
#             st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"--- Starting 3-Step Pipeline (Using {calibration_type} calibration) ---")
            
#             if calibration_type == 'planar':
#                 CALIB_FILE = "./calib_results_planar.txt"
#             else:
#                 CALIB_FILE = "./calib_results_fisheye.txt"
            
#             if not Path(CALIB_FILE).exists():
#                 st.error(f"ERROR: Calibration file not found: {CALIB_FILE}")
#                 st.stop()
            
#             # --- Pass sf_value from the widget ---
#             success, st.session_state.log_output = undistort_images(
#                 INPUT_DIR, OUTPUT_DIR, CALIB_FILE, sf_value, log_area, st.session_state.log_output
#             )
#             if not success:
#                 st.error("Undistortion step failed. Halting.")
#                 st.stop()
            
#             images_to_stitch_dir = OUTPUT_DIR
        
#         else:
#             # Skip undistortion
#             st.session_state.log_output = log_message(log_area, st.session_state.log_output, "--- Starting 2-Step Pipeline (Skipping Undistortion) ---")
#             images_to_stitch_dir = INPUT_DIR

#         # --- Step 2: Write Config ---
#         success, st.session_state.log_output = write_custom_config(
#             CONFIG_FILE, config_values, log_area, st.session_state.log_output
#         )
#         if not success:
#             st.error("Config writing step failed. Halting.")
#             st.stop()
            
#         # --- Step 3: Run Stitching (uses the correct directory) ---
#         success, st.session_state.log_output = run_stitching(
#             images_to_stitch_dir, STITCHER_EXE, log_area, st.session_state.log_output
#         )
#         if not success:
#             st.error("Stitching step failed. Halting.")
#             st.stop()

#         # --- Pipeline Complete ---
#         st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"\n--- Pipeline Complete ---")
#         st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"Total time taken: {time.time()-start:.2f} seconds")
#         st.success("🎉 Pipeline Completed Successfully!")
        
#         # --- 4. Display Result ---
#         final_image_path = Path("out.jpg")
#         if final_image_path.exists():
#             st.image(str(final_image_path), caption="Final Stitched Image", use_column_width=True)
#         else:
#             st.info("Stitching complete. No 'out.jpg' found to display.")

# # --- Run the main app ---
# if __name__ == "__main__":
#     main_app()

"""
Streamlit web application for a 3-step image processing pipeline:
1. (Optional) Undistort omnidirectional images from a user-selected input folder.
2. Write a config.cfg file based on user inputs from the UI.
3. Run an external image stitching executable on the correct set of images.
"""

import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
import time
import sys
import subprocess
import tempfile  # --- MODIFICATION 1: Added for temporary directory ---

import os

# --- Runtime JPEG library fix for Streamlit Cloud ---
def ensure_libjpeg_symlink():
    """
    Ensures libjpeg.so.8 exists (symlinked to libjpeg.so.62) to fix
    'error while loading shared libraries: libjpeg.so.8' on Debian 12.
    """
    lib_dir = "/usr/lib/x86_64-linux-gnu"
    jpeg62 = os.path.join(lib_dir, "libjpeg.so.62")
    jpeg8 = os.path.join(lib_dir, "libjpeg.so.8")
    try:
        if os.path.exists(jpeg62) and not os.path.exists(jpeg8):
            os.symlink(jpeg62, jpeg8)
            print("✅ Created symlink: libjpeg.so.8 → libjpeg.so.62")
        else:
            print("ℹ️ JPEG libraries OK.")
    except Exception as e:
        print("⚠️ Could not create symlink:", e)

# Run once at startup
ensure_libjpeg_symlink()

# Import the OcamModel class
try:
    from ocam_model import OcamModel
except ImportError:
    st.error("ERROR: Could not import OcamModel from ocam_model.py. Make sure it's in the same directory.")
    st.stop()

# --- Core Pipeline Functions (Refactored for Streamlit) ---
# (All helper functions remain the same as your original file)

def log_message(log_area, current_log, message):
    """Helper to append a message to the log display and console."""
    print(message)  # Log to console for debugging
    current_log += message + "\n"
    log_area.code(current_log, language="bash")
    return current_log

def empty_directory(directory_path_str, log_area, current_log):
    """
    Checks for a directory and deletes all *.jpg files inside it.
    """
    dir_path = Path(directory_path_str)
    current_log = log_message(log_area, current_log, f"Checking output directory: {dir_path}")
    if dir_path.exists() and dir_path.is_dir():
        current_log = log_message(log_area, current_log, "Clearing old files from output directory...")
        files_deleted = 0
        try:
            for file_path in dir_path.glob("*.jpg"):
                file_path.unlink()  # Delete the file
                files_deleted += 1
            current_log = log_message(log_area, current_log, f"Deleted {files_deleted} old .jpg files.")
        except Exception as e:
            current_log = log_message(log_area, current_log, f"Warning: Could not delete all files. {e}")
    else:
        current_log = log_message(log_area, current_log, "Output directory not found. It will be created.")
    return current_log

def undistort_images(input_dir, output_dir, calib_file, sf_value, log_area, current_log):
    """
    Loads calibration, finds/undistorts images, and updates Streamlit UI.
    """
    current_log = log_message(log_area, current_log, "--- 1. Starting Image Undistortion ---")
    
    # --- 1a. Load calibration ---
    o = OcamModel()
    current_log = log_message(log_area, current_log, f"Loading calibration from: {calib_file}")
    if o.load_calib(calib_file) != 0:
        st.error(f"ERROR: Failed to load calibration file {calib_file}")
        return False, current_log
    
    # --- 1b. Create output directory (already cleared, just ensure it exists) ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        current_log = log_message(log_area, current_log, f"Output directory set to: {output_dir}")
    except OSError as e:
        st.error(f"ERROR: Could not create directory {output_dir}. {e}")
        return False, current_log

    # --- 1c. Get image list ---
    input_path = Path(input_dir)
    if not input_path.exists():
        st.error(f"ERROR: Input directory not found: {input_dir}")
        return False, current_log
        
    image_paths = sorted(input_path.glob("*.jpg"))
    if not image_paths:
        st.error(f"ERROR: No .jpg images found in directory: {input_dir}")
        return False, current_log
    
    current_log = log_message(log_area, current_log, f"Found {len(image_paths)} images. Starting processing...")
    
    # --- 1d. Loop, undistort, and save ---
    mapx, mapy = None, None
    first_image = True
    
    progress_bar = st.progress(0, "Undistorting images...")
    
    for i, image_path in enumerate(image_paths):
        try:
            src = cv2.imread(str(image_path))
            if src is None:
                current_log = log_message(log_area, current_log, f"Warning: Could not load image: {image_path}")
                continue
            
            if first_image:
                height, width = src.shape[:2]
                current_log = log_message(log_area, current_log, f"Creating undistortion map (LUT) for size {width}x{height}...")
                # Use the sf_value from the UI
                mapx, mapy = o.create_perspective_undistortion_lut(height, width, sf_value)
                first_image = False

            dst_persp = cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)
            output_filename = Path(output_dir) / image_path.name
            cv2.imwrite(str(output_filename), dst_persp)
            
            percent_complete = (i + 1) / len(image_paths)
            progress_bar.progress(percent_complete, f"Processing: {image_path.name}")

        except Exception as e:
            current_log = log_message(log_area, current_log, f"Error processing {image_path.name}: {e}")
    
    progress_bar.empty()
    current_log = log_message(log_area, current_log, "--- Undistortion Complete ---")
    return True, current_log

def write_custom_config(config_path, config_values, log_area, current_log):
    """
    Creates or overwrites the config.cfg file with values from the Streamlit UI.
    """
    current_log = log_message(log_area, current_log, f"--- 2. Writing Custom Config File ({config_path}) ---")
    
    try:
        with open(config_path, 'w') as f:
            for key, value in config_values.items():
                f.write(f"{key} {value}\n")
        
        current_log = log_message(log_area, current_log, "--- Custom Config File Written ---")
        return True, current_log
    except Exception as e:
        st.error(f"ERROR: Could not write config file {config_path}. {e}")
        return False, current_log

def run_stitching(stitched_images_dir, executable_path, log_area, current_log):
    """
    Finds all .jpg files in the directory and runs the stitching executable.
    Streams the output in real-time to the Streamlit log area.
    """
    current_log = log_message(log_area, current_log, f"--- 3. Starting Image Stitching ---")
    
    output_path = Path(stitched_images_dir)
    if not output_path.exists():
        st.error(f"ERROR: Directory for stitching not found: {stitched_images_dir}")
        return False, current_log
        
    images_to_stitch = sorted(output_path.glob("*.jpg"))

    if not images_to_stitch:
        st.warning(f"Warning: No .jpg images found in {stitched_images_dir} to stitch.")
        return True, current_log

    cmd = [executable_path] + [str(p) for p in images_to_stitch]
    current_log = log_message(log_area, current_log, f"Running command: {' '.join(cmd[:4])}... ({len(cmd)-1} files total)\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                current_log += line
                log_area.code(current_log, language="bash")

        process.wait()

        current_log += f"\n--- Image Stitching Finished (Return Code: {process.returncode}) ---"
        log_area.code(current_log, language="bash")

        if process.returncode != 0:
            st.error(f"ERROR: Image stitching executable failed with code {process.returncode}")
            return False, current_log
        
        return True, current_log

    except FileNotFoundError:
        st.error(f"ERROR: Executable not found at '{executable_path}'. Make sure it's executable (chmod +x {executable_path}).")
        return False, current_log
    except Exception as e:
        st.error(f"An error occurred while running {executable_path}: {e}")
        return False, current_log

# --- Streamlit UI Application ---

def main_app():
    st.title("📸 Image Undistortion & Stitching Pipeline")

    # --- 1. Sidebar for all Configuration ---
    st.sidebar.title("Pipeline Configuration")

    # --- Main Settings ---
    st.sidebar.header("Pipeline Settings")
    
    # --- MODIFICATION 2: Replaced text_input with file_uploader ---
    uploaded_files = st.sidebar.file_uploader(
        "Upload Input Images",
        type=["jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload all the .jpg images you want to process."
    )
    
    st.sidebar.header("Step 1: Undistortion")
    do_undistort = st.sidebar.toggle("Run Undistortion Step", value=True, help="If disabled, the stitcher will run on the original images from the Input Image Folder.")
    calibration_type = st.sidebar.radio(
        "Calibration Type", 
        ["planar", "fisheye"], 
        horizontal=True, 
        disabled=not do_undistort,
        help="Which calibration file to use. Only active if 'Run Undistortion' is ON."
    )
    
    sf_value = st.sidebar.number_input(
        "Zoom Factor (SF_VALUE)",
        value=4.0,
        min_value=0.1,
        step=0.1,
        format="%.1f",
        disabled=not do_undistort,
        help="Zoom factor for perspective undistortion. Only active if 'Run Undistortion' is ON."
    )

    # --- Config File Settings (No changes here) ---
    st.sidebar.header("Step 2: Stitching Config")
    
    config_values = {}

    with st.sidebar.expander("General Modes"):
        config_values["CYLINDER"] = 1 if st.toggle("CYLINDER", value=False) else 0
        config_values["ESTIMATE_CAMERA"] = 1 if st.toggle("ESTIMATE_CAMERA", value=False) else 0
        config_values["TRANS"] = 1 if st.toggle("TRANS", value=True) else 0
        config_values["ORDERED_INPUT"] = 1 if st.toggle("ORDERED_INPUT", value=True) else 0
        config_values["CROP"] = 1 if st.toggle("CROP", value=False) else 0
        config_values["MAX_OUTPUT_SIZE"] = st.number_input("MAX_OUTPUT_SIZE", value=8000, step=100)
        config_values["LAZY_READ"] = 1 if st.toggle("LAZY_READ", value=True) else 0
        config_values["FOCAL_LENGTH"] = st.number_input("FOCAL_LENGTH", value=10.0, format="%.2f")

    with st.sidebar.expander("Keypoint Parameters"):
        config_values["SIFT_WORKING_SIZE"] = st.number_input("SIFT_WORKING_SIZE", value=1000, step=100)
        config_values["NUM_OCTAVE"] = st.number_input("NUM_OCTAVE", value=4, step=1)
        config_values["NUM_SCALE"] = st.number_input("NUM_SCALE", value=7, step=1)
        config_values["SCALE_FACTOR"] = st.number_input("SCALE_FACTOR", value=1.4142135623, format="%.10f")
        config_values["GAUSS_SIGMA"] = st.number_input("GAUSS_SIGMA", value=1.4142135623, format="%.10f")
        config_values["GAUSS_WINDOW_FACTOR"] = st.number_input("GAUSS_WINDOW_FACTOR", value=5, step=1)
        config_values["CONTRAST_THRES"] = st.number_input("CONTRAST_THRES", value=3e-10, format="%e")
        config_values["JUDGE_EXTREMA_DIFF_THRES"] = st.number_input("JUDGE_EXTREMA_DIFF_THRES", value=1e-10, format="%e")
        config_values["EDGE_RATIO"] = st.number_input("EDGE_RATIO", value=100, step=1)
        config_values["PRE_COLOR_THRES"] = st.number_input("PRE_COLOR_THRES", value=5e-2, format="%e")
        config_values["CALC_OFFSET_DEPTH"] = st.number_input("CALC_OFFSET_DEPTH", value=4, step=1)
        config_values["OFFSET_THRES"] = st.number_input("OFFSET_THRES", value=1.0, step=0.1, format="%.1f")

    with st.sidebar.expander("Descriptor & Matching"):
        config_values["ORI_RADIUS"] = st.number_input("ORI_RADIUS", value=4.5, format="%.1f")
        config_values["ORI_HIST_SMOOTH_COUNT"] = st.number_input("ORI_HIST_SMOOTH_COUNT", value=2, step=1)
        config_values["DESC_HIST_SCALE_FACTOR"] = st.number_input("DESC_HIST_SCALE_FACTOR", value=3, step=1)
        config_values["DESC_INT_FACTOR"] = st.number_input("DESC_INT_FACTOR", value=512, step=1)
        config_values["MATCH_REJECT_NEXT_RATIO"] = st.number_input("MATCH_REJECT_NEXT_RATIO", value=0.8, format="%.1f")
        config_values["RANSAC_ITERATIONS"] = st.number_input("RANSAC_ITERATIONS", value=8000, step=100)
        config_values["RANSAC_INLIER_THRES"] = st.number_input("RANSAC_INLIER_THRES", value=3.5, format="%.1f")
        config_values["INLIER_IN_MATCH_RATIO"] = st.number_input("INLIER_IN_MATCH_RATIO", value=0.1, format="%.2f")
        config_values["INLIER_IN_POINTS_RATIO"] = st.number_input("INLIER_IN_POINTS_RATIO", value=0.04, format="%.2f")
    
    with st.sidebar.expander("Optimization & Blending"):
        config_values["STRAIGHTEN"] = 1 if st.toggle("STRAIGHTEN", value=True) else 0
        config_values["SLOPE_PLAIN"] = st.number_input("SLOPE_PLAIN", value=8e-3, format="%e")
        config_values["LM_LAMBDA"] = st.number_input("LM_LAMBDA", value=5, step=1)
        config_values["MULTIPASS_BA"] = st.selectbox("MULTIPASS_BA", [0, 1, 2], index=2)
        config_values["MULTIBAND"] = st.number_input("MULTIBAND", value=10, step=1)

    # --- 2. Main Page Layout ---
    
    log_area = st.code("Waiting to start. Configure settings in the sidebar and click 'Run Pipeline'.", language="bash")
    
    # --- MODIFICATION 3: Updated "Run Pipeline" logic ---
    if st.sidebar.button("🚀 Run Pipeline", type="primary"):
        
        # --- 0. Check for uploaded files ---
        if not uploaded_files:
            st.warning("Please upload images to process.")
            st.stop()
            
        # --- 3. Run Pipeline Logic ---
        st.session_state.log_output = "" # Reset log
        start = time.time()
        
        # --- Define Paths ---
        OUTPUT_DIR = "./result"
        CONFIG_FILE = "./config.cfg"
        STITCHER_EXE = "./image-stitching"
        
        # --- Create a Temporary Directory to store uploaded files ---
        with tempfile.TemporaryDirectory() as tmpdir:
            st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"Created temporary input directory: {tmpdir}")
            
            # --- Save uploaded files to the temp directory ---
            for f in uploaded_files:
                try:
                    file_path = os.path.join(tmpdir, f.name)
                    with open(file_path, "wb") as out_f:
                        out_f.write(f.getbuffer())
                except Exception as e:
                    st.error(f"Error saving uploaded file {f.name}: {e}")
                    st.stop()
            
            st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"Saved {len(uploaded_files)} images to {tmpdir}.")

            # --- Set INPUT_DIR to our new temporary directory ---
            INPUT_DIR = tmpdir
            images_to_stitch_dir = ""

            # --- Call the empty_directory function ---
            st.session_state.log_output = empty_directory(OUTPUT_DIR, log_area, st.session_state.log_output)

            # --- Step 1: (Conditional) Undistortion ---
            if do_undistort:
                st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"--- Starting 3-Step Pipeline (Using {calibration_type} calibration) ---")
                
                if calibration_type == 'planar':
                    CALIB_FILE = "./calib_results_planar.txt"
                else:
                    CALIB_FILE = "./calib_results_fisheye.txt"
                
                if not Path(CALIB_FILE).exists():
                    st.error(f"ERROR: Calibration file not found: {CALIB_FILE}")
                    st.stop()
                
                # --- Pass sf_value from the widget ---
                success, st.session_state.log_output = undistort_images(
                    INPUT_DIR, OUTPUT_DIR, CALIB_FILE, sf_value, log_area, st.session_state.log_output
                )
                if not success:
                    st.error("Undistortion step failed. Halting.")
                    st.stop()
                
                images_to_stitch_dir = OUTPUT_DIR
            
            else:
                # Skip undistortion
                st.session_state.log_output = log_message(log_area, st.session_state.log_output, "--- Starting 2-Step Pipeline (Skipping Undistortion) ---")
                images_to_stitch_dir = INPUT_DIR

            # --- Step 2: Write Config ---
            success, st.session_state.log_output = write_custom_config(
                CONFIG_FILE, config_values, log_area, st.session_state.log_output
            )
            if not success:
                st.error("Config writing step failed. Halting.")
                st.stop()
                
            # --- Step 3: Run Stitching (uses the correct directory) ---
            success, st.session_state.log_output = run_stitching(
                images_to_stitch_dir, STITCHER_EXE, log_area, st.session_state.log_output
            )
            if not success:
                st.error("Stitching step failed. Halting.")
                st.stop()

            # --- Pipeline Complete ---
            st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"\n--- Pipeline Complete ---")
            st.session_state.log_output = log_message(log_area, st.session_state.log_output, f"Total time taken: {time.time()-start:.2f} seconds")
            st.success("🎉 Pipeline Completed Successfully!")
        
        # --- End of the 'with tempfile.TemporaryDirectory()' block ---
        # The temporary directory and its contents are now automatically deleted.
        
        # --- 4. Display Result ---
        final_image_path = Path("out.jpg")
        if final_image_path.exists():
            st.image(str(final_image_path), caption="Final Stitched Image", use_column_width=True)
            
            # --- MODIFICATION 4: Add a download button ---
            with open(str(final_image_path), "rb") as file:
                st.download_button(
                    label="Download Stitched Image",
                    data=file,
                    file_name="stitched_result.jpg",
                    mime="image/jpeg"
                )
        else:
            st.info("Stitching complete. No 'out.jpg' found to display.")

# --- Run the main app ---
if __name__ == "__main__":
    main_app()