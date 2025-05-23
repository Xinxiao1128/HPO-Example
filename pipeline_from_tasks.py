from clearml import Task
from clearml.automation import PipelineController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration - using same queue for everything
EXECUTION_QUEUE = "pipeline"

def run_pipeline():
    pipe = PipelineController(
        name="Pipeline demo", 
        project="Agri-Pest-Detection", 
        version="0.0.1", 
        add_pipeline_tags=False
    )

    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # Step 1 - Dataset
    pipe.add_step(
        name="stage_data",
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 1 - Load Uncompressed Pest Image Dataset",
        execution_queue=EXECUTION_QUEUE
    )

    # Step 2 - Preprocessing
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 2 - Preprocessing (artifact version)",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/test_size": 0.25,
            "General/random_state": 42
        }
    )

    # Step 3 - Initial training
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 3 - Model Training",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.id}",  # ✅ FIXED
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_epochs": 20,
            "General/batch_size": 16,
            "General/learning_rate": 1e-3,
            "General/weight_decay": 1e-5
        }
    )

    # Step 4 - HPO
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_train", "stage_process", "stage_data"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 4 - HPO: Train Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.id}",  # ✅ FIXED
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_trials": 4,
            "General/time_limit_minutes": 20,
            "General/run_as_service": False,
            "General/dataset_task_id": "${stage_data.id}",
            "General/base_train_task_id": "${stage_train.id}"
        }
    )

    # Step 5 - Final model training
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo", "stage_process"],
        base_task_project="Agri-Pest-Detection",
        base_task_name="Step 5 - Final Model Training",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.id}",  # ✅ FIXED
            "General/hpo_task_id": "${stage_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE
        }
    )

    logger.info("Starting pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally()
    logger.info("Pipeline started successfully")

if __name__ == "__main__":
    run_pipeline()
