import pandas as pd
import subprocess
import os

from tqdm import tqdm
import configargparse

def parse_config():
    parser = configargparse.ArgumentParser(
                        prog='Multi-View Diffusion',
                        description='Generate texture given mesh and texture prompt',
                        epilog='Refer to https://arxiv.org/abs/2311.12891 for more details')
    # File Config
    parser.add_argument('--prompt', type=str, required=False)
    options = parser.parse_args()

    return options


max_hits = 2
style_prompt = None
style_prompt = "Christmas style"

objects_path = "Objaverse_Objects.csv"
meshes_path = "final_objects"

def main():
    global style_prompt
    opt = parse_config()
    if opt.prompt is not None:
        style_prompt = opt.prompt
    # Read the CSV and extract the list of uids
    df = pd.read_csv(objects_path)
    uid_list = df['uid'].tolist()

    # Loop through the uid list and run the experiment for each uid
    for uid in tqdm(uid_list):
        config_path = f"{meshes_path}/{uid}/config.yaml"
        
        if not os.path.exists(config_path):
            print(f"Config file missing: {config_path}")
            continue
        
        # Construct the command with max_hits
        command = f"python run_experiment.py --config {config_path}"
        if style_prompt is not None:
            command += f" --style_prompt \"{style_prompt}\""
        command += f" --max_hits {max_hits}"
        
        # Run the command
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)

if __name__ == '__main__':
    main()
