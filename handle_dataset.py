
from Utils import remove_from_h5, merge_h5_datasets, merge_json_summaries
import os


# remove_from_h5(
#     base_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16'),
#     # base_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (fixed_deployment)'),
#     filename='delay.h5',
#     dataset_name='hrhm1oqz'
# )


merge_h5_datasets(
    base_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (expert3_10-90)_base'),
    new_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (expert3_10-90_model_8601600_steps)'),
    filename='delay.h5',
    overwrite=True  # Set to False to avoid overwriting
)


# merge_json_summaries(
#     base_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (expert3_10-90)_base'),
#     new_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (expert3_10-90_model_8601600_steps)'),
#     base_filename='summary.json',
#     new_filename='summary.json',
#     overwrite=True  # Set False to skip existing keys
# )