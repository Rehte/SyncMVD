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


max_hits = [2, 4]
style_prompt = None
style_prompts = [
    None,
    "Halloween style on the outside and christmas style inside",
    "Cyberpunk style",
    "Luxury style",
    "Festival style",
    "Underwater world style",
    "Carnival style"
    "Halloween style",
    "Christmas style",
]

category_prompts = {
    "car": [

    ],
    "cup": [

    ],
    "hat": [

    ],
    "house": [ # also cabin

    ],
    "tent": [

    ],
    "ring": [

    ],
    "crown": [

    ]
}

run_multiple_style_prompts = True

objects_path = "Objects_Run.csv"
objects_path = "Objaverse_Objects.csv"
meshes_path = "final_objects3"

def run_batch(uid_list, style_prompt=None, max_hits=2):
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
        if max_hits > 2:
            command += " --disable_base_hit"
        
        # Run the command
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)

def main():
    global style_prompt, style_prompts, max_hits, run_multiple_style_prompts
    opt = parse_config()
    if opt.prompt is not None:
        style_prompt = opt.prompt
    # Read the CSV and extract the list of uids
    df = pd.read_csv(objects_path)
    uid_list = df['uid'].tolist()

    # Loop through the uid list and run the experiment for each uid
    if run_multiple_style_prompts:
        for max_hit in max_hits:
            for style_prompt in style_prompts:
                run_batch(uid_list, style_prompt, max_hit)
    else:
        run_batch(uid_list, style_prompt, max_hits)

if __name__ == '__main__':
    main()
