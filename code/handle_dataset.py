
from utils import remove_from_h5, merge_h5_datasets, merge_json_summaries
import os


# remove_from_h5(
#     base_dir=os.path.join(os.getcwd(), 'results_for_tuning_structured/general[10,90]/30-16'),
#     # base_dir=os.path.join(os.getcwd(), 'Results/Simulation/30-16 (fixed_deployment)'),
#     filename='delay.h5',
#     dataset_names=['4xwrtcvd','w581iqyg','r1eoj88r', 'ttmy7wk1','az4v9nku']
# )


merge_h5_datasets(
    base_dir=os.path.join(os.getcwd(), 'results_for_tuning_structured/general[10,90]/30-16'),
    new_dir=os.path.join(os.getcwd(), 'results_for_tuning_structured_new/general[10,90]/30-16'),
    filename='delay.h5',
    overwrite=True  # Set to False to avoid overwriting
)


# merge_json_summaries(
#     base_dir=os.path.join(os.getcwd(), 'results_for_tuning_structured/general[10,90]/30-16'),
#     new_dir=os.path.join(os.getcwd(), 'results_for_tuning_structured_new/general[10,90]/30-16'),
#     base_filename='summary.json',
#     new_filename='summary.json',
#     overwrite=True  # Set False to skip existing keys
# )