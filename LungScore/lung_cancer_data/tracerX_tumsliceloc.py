import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
import numpy as np
import pandas as pd


def check_num_of_blankedslices(loc):
    data = []  # List to store the results for Excel
    j = 0  # Counter for total patients with processed scans

    for pat in os.listdir(loc):
        scan = np.load(os.path.join(loc, pat))  # Load each scan
        num_of_blanked_slice = 0
        first_blanked_slice = None  # Track the first blank slice index
        blank_slice_threshold = 10  # Define the number of consecutive blank slices to ignore
        blank_slices_count = 0  # Count of consecutive blank slices at the beginning

        for num_slices in range(scan.shape[0]):
            one_slice = scan[num_slices]

            # Check if the current slice is blank
            if np.all(one_slice == -1024):
                blank_slices_count += 1

                if first_blanked_slice is None:
                    first_blanked_slice = num_slices  # Mark the first blank slice
                    num_of_blanked_slice += 1  # Count it as a blank slice
                else:
                    diff = num_slices - first_blanked_slice
                    if diff < 28:
                        num_of_blanked_slice += 1  # Count this blank slice if within range
            else:
                if first_blanked_slice is not None:
                    if blank_slices_count < blank_slice_threshold:
                        first_blanked_slice = None
                        num_of_blanked_slice = 0  # Reset count
                    else:
                        break  # Exit loop after processing valid data

        # Check if we have exactly 28 blank slices
        if num_of_blanked_slice == 28:
            center_of_blanking_location = first_blanked_slice + 13
            center_group = determine_center_group(center_of_blanking_location)
            data.append([pat[:7], center_of_blanking_location, center_group])  # Store patient ID, location, and group
        else:
            print(pat)
            print(num_of_blanked_slice)
            j += 1

    # Create a DataFrame and save to Excel after processing all patients
    df = pd.DataFrame(data, columns=['Patient Name', 'Blank Center Location', 'Center Group'])
    df.to_excel('/mnt/data6/DeepPY/ai_lung_damage/trx_28removedslices_patient_blank_slices_loc.xlsx', index=False)

    print(f'Total patients with num of blank slices != 28 : {j}')

def determine_center_group(location):
    """Determine the center group based on the center location."""
    if location < 30:
        return 0
    elif location < 60:
        return 1
    elif location < 90:
        return 2
    else:
        return None  # Return None for locations outside the defined groups


###############Example Usage####################
check_num_of_blankedslices('/mnt/data6/DeepPY/ai_lung_damage/data/trx_wl_tumorsliceremovedmax/')
