import os
import subprocess

base_path = 'final_objects3'

def save_mesh_as_glb(mesh_path, file_path):
    """
    Save the given mesh as a GLB file.
    """
    if not os.path.exists(file_path):
        # Load the mesh with textures and convert to GLB
        command = f"obj2gltf -i {mesh_path} -o {file_path}"
        subprocess.run(command, shell=True)
        print(f"Converted {mesh_path} to {file_path}")
    else:
        print(f"GLB already exists for {mesh_path}, skipping.")

def process_results(results_path):
    """
    Process all .obj files in the results folder, converting them to GLB.
    """
    for file_name in os.listdir(results_path):
        if file_name.endswith('.obj'):
            obj_file_path = os.path.join(results_path, file_name)
            glb_file_path = os.path.splitext(obj_file_path)[0] + '.glb'
            save_mesh_as_glb(obj_file_path, glb_file_path)

def main():
    for mesh_folder in os.listdir(base_path):
        mesh_folder_path = os.path.join(base_path, mesh_folder)
        
        if os.path.isdir(mesh_folder_path):
            for run_folder in os.listdir(mesh_folder_path):
                run_folder_path = os.path.join(mesh_folder_path, run_folder)
                
                if os.path.isdir(run_folder_path):
                    results_path = os.path.join(run_folder_path, 'results')
                    
                    if os.path.exists(results_path) and os.path.isdir(results_path):
                        process_results(results_path)

if __name__ == "__main__":
    main()