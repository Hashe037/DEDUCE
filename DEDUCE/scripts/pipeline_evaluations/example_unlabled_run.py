from deduce.dataset_evaluation_pipeline import DatasetEvaluationPipeline
from deduce.utils.export import filter_predictions
import os

# Initialize pipeline with config
pipeline = DatasetEvaluationPipeline("../configs/caltech_runs/relabel_transtest_1/test1_uavempty.ini")

# Load dataset
dataset = pipeline.load_dataset()

# Run zero-shot prediction
results = pipeline.predict(dataset)

# Evaluate results
evaluation = pipeline.evaluate(results)
pipeline.visualize(results, dataset)

# Save results using enhanced saver
save_path = pipeline.config_manager.get('EVALUATION', {}).get('output_path')

# Save predictions
descriptor_list = ["day_night"]
# descriptor_list = ["day_night", "nodamage_damage", "sharp_blurry", "noveg_veg",
#                     "clear_fog", "noflare_flare", "lowdense_highdense"]

for descriptor in descriptor_list:
    each_desc = descriptor.split('_')
    confidence_results = filter_predictions(
        os.path.join(save_path,'filename_margins_results.json'),
        descriptor=descriptor,
        prediction=each_desc[0],
        min_margin=0.010,
        max_margin=0.025, 
        save_path=os.path.join(save_path,f"{each_desc[0]}_filenames.json")
    )
    confidence_results = filter_predictions(
        os.path.join(save_path,'filename_margins_results.json'),
        descriptor=descriptor,
        prediction=each_desc[1],
        min_margin=0.010,
        max_margin=0.025, 
        save_path=os.path.join(save_path,f"{each_desc[1]}_filenames.json")
    )