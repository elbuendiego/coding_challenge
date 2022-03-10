import mlflow
import os
import hydra
import json

from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"]=str(1)

# Read configuration through hydra
@hydra.main(config_name='config')
def run_components(config: DictConfig):

    # Path at root of MLflow project and files
    root_path = hydra.utils.get_original_cwd()
                                            

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        "main",
        parameters={
            "file_path": os.path.join(root_path,
                                     "Data",
                                      config["data"]["file_name"]
                                      ),
            "out_path": os.path.join(root_path,
                                     "Data",
                                     config["data"]["out_name"],
                                     )
        },
    )

    # Now create file with random forest parameters
    rf_params = os.path.abspath("random_forest_config.json")

    with open(rf_params, "w+") as fp:
        json.dump(dict(config["random_forest"]), fp)

    _ = mlflow.run(
        os.path.join(root_path, "train_model"),
        "main",
        parameters={
            "train_data": os.path.join(root_path,
                                    "Data",
                                    config["data"]["out_name"],
                                    ),
            "rf_params": rf_params
        },
    )



if __name__ == "__main__":
    run_components()
