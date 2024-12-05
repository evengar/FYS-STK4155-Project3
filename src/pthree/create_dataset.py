import numpy as np
import pandas as pd
import git
import os

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir


def create_full_dataset(directory, save_as):
    """Function to combine tsv files into a full dataset."""
    path = f"{directory}{save_as}.csv"
    if os.path.isfile(path):
        print("Full ecotaxa dataset exist!")
        return
    
    dataset = []

    for file in os.scandir(directory):
        if file.is_file():
            filename = os.path.join(directory, file)
            data = pd.read_csv(filename, sep="\t")
            data.drop([0], inplace=True)
            dataset.append(data)
        else:
            continue
    
    full_dataset = pd.concat(dataset, ignore_index=True)
    full_dataset.to_csv(f"{directory}{save_as}.csv")


def feature_selection_ecotaxa(path_file):
    """Work in progress, need to decide on features to use."""
    dataset = pd.read_csv(path_file, index_col=0)

    dataset["target_name"] = dataset.apply(lambda x: x["img_file_name"].split("/")[0], axis=1)
    labels = np.unique(dataset.apply(lambda x: x["target_name"], axis=1)).tolist()
    dataset["target"] = dataset.apply(lambda x: labels.index(x["target_name"]), axis=1)

    start_idx = dataset.columns.to_list().index("object_label")
    end_idx = dataset.columns.to_list().index("object_date_end")
    features = dataset.columns.to_list()[start_idx:end_idx]

    X = dataset[[c for c in dataset.columns if c in features]].to_numpy()
    y = dataset["target"].to_numpy()

    return X, y


if __name__ == '__main__':
    directory = f"{PATH_TO_ROOT}/data/metadata/"

    create_full_dataset(directory, "ecotaxa_full")
    
    path_file = f"{directory}ecotaxa_full.csv"
    X, y = feature_selection_ecotaxa(path_file)
    


"""Metadata feature columns:
img_file_name
img_rank
object_id
object_lat
object_lon
object_date
object_time
object_link
object_depth_min
object_depth_max
object_annotation_status
object_annotation_person_name
object_annotation_person_email
object_annotation_date
object_annotation_time
object_annotation_category
object_annotation_hierarchy
complement_info
object_label
object_width
object_height
object_bx
object_by
object_circ.
object_area_exc
object_area
object_%area
object_major
object_minor
object_y
object_x
object_convex_area
object_perim.
object_elongation
object_perimareaexc
object_perimmajor
object_circex
object_angle
object_bounding_box_area
object_eccentricity
object_equivalent_diameter
object_euler_number
object_extent
object_local_centroid_col
object_local_centroid_row
object_solidity
object_meanhue
object_meansaturation
object_meanvalue
object_stdhue
object_stdsaturation
object_stdvalue
object_date_end
object_time_end
object_lat_end
object_lon_end
sample_id
sample_dataportal_descriptor
sample_project
sample_ship
sample_operator
sample_sampling_gear
sample_concentrated_sample_volume
sample_dilution_factor
sample_gear_net_opening
sample_total_volume
sample_uuid
process_id
process_pixel
process_source
process_commit
process_datetime
process_uuid
process_1st_operation
process_2nd_operation
process_3rd_operation
process_4th_operation
process_5th_operation
process_6th_operation
process_7th_operation
acq_id
acq_instrument
acq_instrument_id
acq_celltype
acq_minimum_mesh
acq_maximum_mesh
acq_volume
acq_imaged_volume
acq_magnification
acq_fnumber_objective
acq_camera
acq_nb_frame
acq_software
acq_local_datetime
acq_camera_resolution
acq_camera_iso
acq_camera_shutter_speed
acq_uuid
objid
processid_internal
acq_id_internal
sample_id_internal
classif_id
classif_who
classif_auto_id
classif_auto_name
classif_auto_score
classif_auto_when
object_random_value
object_sunpos
sample_lat
sample_long
"""