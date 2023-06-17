import argparse
import os

import git
import yaml
from determined.common.experimental import experiment
from determined.common.experimental.experiment import ExperimentState
from determined.experimental import Determined

# =====================================================================================


class DeterminedClient(Determined):
    def __init__(self, master, user, password):
        super().__init__(master=master, user=user, password=password)

    def continue_experiment(self, config, parent_id, checkpoint_uuid):
        config["searcher"]["source_checkpoint_uuid"] = checkpoint_uuid

        resp = self._session.post(
            "/api/v1/experiments",
            json={
                "activate": True,
                "config": yaml.safe_dump(config),
                "parentId": parent_id,
            },
        )

        exp_id = resp.json()["experiment"]["id"]
        exp = experiment.ExperimentReference(exp_id, self._session)

        return exp


# =====================================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Determined AI Experiment Runner")

    parser.add_argument(
        "--pach_config",
        type=str,
        help="Pachyderm's configuration file",
    )

    parser.add_argument(
        "--pach_project_name",
        type=str,
        help="Pachyderm's project name",
    )

    parser.add_argument(
        "--repo",
        type=str,
        help="Name of the Pachyderm's repository containing the dataset",
    )

    parser.add_argument(
        "--branch",
        type=str,
        help="Name of the Pachyderm's repository's branch containing the dataset",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Determined's experiment configuration file",
    )

    parser.add_argument(
        "--git-url",
        type=str,
        help="Git URL of the repository containing the model code",
    )

    parser.add_argument(
        "--git-ref",
        type=str,
        help="Git Commit/Tag/Branch to use",
    )

    parser.add_argument(
        "--sub-dir",
        type=str,
        help="Subfolder to experiment files",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Name of the model on DeterminedAI to create/update",
    )

    parser.add_argument(
        "--work-dir",
        type=str,
        help="Directory containing the experiment code to be submitted to DeterminedAI.",
    )

    return parser.parse_args()


# =====================================================================================


def clone_code(repo_url, ref, dir):
    print(f"Cloning code from: {repo_url}@{ref} --> {dir}")
    if os.path.isdir(dir):
        repo = git.Repo(dir)
        repo.remotes.origin.fetch()
    else:
        repo = git.Repo.clone_from(repo_url, dir)
    repo.git.checkout(ref)


# =====================================================================================


def read_config(conf_file):
    print(f"Reading experiment config file: {conf_file}")
    config = {}
    with open(conf_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


# =====================================================================================


def setup_config(config_file, project, repo, branch, input_commit, pipeline, job_id):
    config = read_config(config_file)
    config["data"]["pachyderm"]["host"] = os.getenv("PACHD_LB_SERVICE_HOST")
    config["data"]["pachyderm"]["port"] = os.getenv("PACHD_LB_SERVICE_PORT")
    config["data"]["pachyderm"]["project"] = project
    config["data"]["pachyderm"]["repo"] = repo
    # config["data"]["pachyderm"]["pipeline_input_name"]
    config["data"]["pachyderm"]["branch"] = branch
    config["data"]["pachyderm"]["commit"] = input_commit
    config["data"]["pachyderm"]["job_id"] = job_id
    config["data"]["pachyderm"]["token"] = os.getenv("PAC_TOKEN")

    config["labels"] = [repo, job_id, pipeline]

    return config


# =====================================================================================

def create_client(user=None):
    if user is None:
        client = DeterminedClient(master=os.getenv("DET_MASTER"),
                                  user=os.getenv("DET_USER"),
                                  password=os.getenv("DET_PASSWORD"),
                                  )
    else:
        client = DeterminedClient(master=os.getenv("DET_MASTER"),
                                  user=user,
                                  password=os.getenv("DET_PASSWORD"),
                                  )
    return client


# =====================================================================================


def execute_experiment(
    client, configfile, code_path, checkpoint, pach_version=None
):
    try:
        if checkpoint is None:
            parent_id = None
            configfile["data"]["pachyderm"]["previous_commit"] = None
            exp = client.create_experiment(configfile, code_path)
        else:
            parent_id = checkpoint.training.experiment_id
            configfile["data"]["pachyderm"]["previous_commit"] = pach_version
            exp = client.continue_experiment(configfile, parent_id, checkpoint.uuid)

        print(f"Created experiment with id='{exp.id}' (parent_id='{parent_id}'). Waiting for its completion...")
        state = exp.wait()
        
        print(f"Experiment with id='{exp.id}' ended with the following state: {state}")
        if state == ExperimentState.COMPLETED:
            return exp
        else:
            return None
        
    except AssertionError:
        print("Experiment exited with abnormal state")
        return None


# =====================================================================================


def run_experiment(client, configfile, code_path, model):
    version = model.get_version()

    if version is None:
        print("Creating a new experiment on DeterminedAI...")
        return execute_experiment(client, configfile, code_path, None)
    else:
        print("Continuing experiment on DeterminedAI...")
        return execute_experiment(client, configfile, None, version.checkpoint, version.name)


# =====================================================================================


def get_checkpoint(exp):
    try:
        return exp.top_checkpoint()
    except AssertionError:
        return None


# =====================================================================================
def get_or_create_model(client, model_name, pipeline, repo, workspace):
    # Retrieve the models from the DeterminedAI registry into a list.
    models = client.get_models(name=model_name)

    if len(models) > 0:
        print(
            f"Model already present in the determinedAI model registry. Updating it : {model_name}")
        model = client.get_models(name=model_name)[0]
    else:
        print(
            f"Creating a new model entry at the determinedAI's model registry. The model name is: {model_name}")
        model = client.create_model(name=model_name,
                                    labels=[pipeline, repo],  # they appear as tags on the determinedAI dashboard
                                    metadata={"pipeline": pipeline,
                                              "repository": repo},
                                    workspace_name=workspace,
                                    )
    return model


# =====================================================================================


def register_checkpoint(checkpoint, model, job_id):
    print(f"Registering/Upload checkpoint to the model registry : {model.name}")
    version = model.register_version(checkpoint.uuid)
    version.set_name(job_id)
    version.set_notes("Job_id/commit_id = " + job_id)

    checkpoint.download("/pfs/out/checkpoint")
    print("Checkpoint registered and downloaded to output repository")


# =====================================================================================


def write_model_info(file, model_name, model_version, pipeline, repo):
    print(f"Writing model information to file: {file}")

    model = dict()
    model["name"] = model_name
    model["version"] = model_version
    model["pipeline"] = pipeline
    model["repo"] = repo

    with open(file, "w") as stream:
        try:
            yaml.safe_dump(model, stream)
        except yaml.YAMLError as exc:
            print(exc)


# =====================================================================================


def main():
    # --- Retrieve useful info from environment

    job_id = os.getenv("PACH_JOB_ID")
    pipeline = os.getenv("PPS_PIPELINE_NAME")
    args = parse_args()

    # Use latest commit from pachyderm pipeline input data repo
    original_pachyderm_config = read_config(args.pach_config)
    input_commit_env_name = original_pachyderm_config["input"]["pfs"]["name"]+"_COMMIT"
    input_commit = os.getenv(input_commit_env_name)
    print(f"Starting pipeline: name='{pipeline}', 'project_name'='{args.pach_project_name}', repo='{args.repo}', 'branch='{args.branch}, 'input_repo_commit:'{input_commit}', job_id='{job_id}'")

    workdir = args.work_dir
    config_file = os.path.join(workdir, args.config)

    # # --- Read and setup experiment config file. Then, run experiment
    config = setup_config(config_file, args.pach_project_name,
                          args.repo, args.branch, input_commit, pipeline, job_id)

    # create determinedAI client
    if config["workspace"] == "khanghua.boon":
        det_client = create_client(user=config["workspace"])
    else:
        det_client = create_client()

    # retrieve or create the model on the determinedAI model registry. pipeline and args.repo are metadata added to the model registry. Only args.model is required.
    model = get_or_create_model(det_client, args.model, pipeline, args.repo, config["workspace"])
    
    # Submit experiment to mldm platform and return the experiment metadata
    exp = run_experiment(det_client, config, workdir, model)
    if exp is None:
        print("Aborting pipeline as experiment did not succeed")
        return

    # --- Get best checkpoint from experiment. It may not exist if the experiment did not succeed
    checkpoint = get_checkpoint(exp)
    if checkpoint is None:
        print("No checkpoint found (probably there was no data). Aborting pipeline")
        return

    # --- Now, register checkpoint on model and download it
    register_checkpoint(checkpoint, model, job_id)
    write_model_info("/pfs/out/model-info.yaml", args.model, job_id, pipeline, args.repo)

    # print("workdir: ", workdir)
    # print('original pachyderm config_file: ', original_pachyderm_config)
    # print('final usable config file: ', config)
    # print(os.environ)
    print(f"Ending pipeline: name='{pipeline}', 'project_name'='{args.pach_project_name}', repo='{args.repo}', 'branch='{args.branch}, 'input_repo_commit:'{input_commit}', job_id='{job_id}'")


# =====================================================================================


if __name__ == "__main__":
    main()
