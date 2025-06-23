import os
import pandas as pd
import json
import base64
import subprocess

CLUSTER_COLORS = {
    "Absolute probability": "#2E8B57",
    "In between": "#FF8C00",
    "Zero probability": "#DC143C",
}


def modify_image_description(filename, description):
    """Call the modify_description.py script"""
    try:
        result = subprocess.run(
            ["python", "modify_description.py", filename, description],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            return True, f"Successfully modified description for {filename}"
        else:
            return False, f"Error: {result.stderr or result.stdout}"
    except Exception as e:
        return False, f"Exception occurred: {str(e)}"


def load_data(
    prob_file: str = "step2a_result.txt",
    image_folder: str = "/home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/data/stanford-40-actions/JPEGImages",
):
    if not os.path.exists(prob_file):
        return pd.DataFrame()

    image_files, probs = [], []
    with open(prob_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                file_part, json_part = line.split(";", 1)
                name_raw = file_part.split(" ", 1)[1].strip()
                filename = name_raw[5:] if name_raw.startswith("file-") else name_raw
                data = json.loads(json_part)
                prob = float(data.get("entailment_probability", 0.0))
            except:
                continue
            image_files.append(filename)
            probs.append(prob)

    df = pd.DataFrame({"filename": image_files, "prob": probs})

    def assign_cluster(p):
        if p == 0.0:
            return "Zero probability"
        if p == 1.0:
            return "Absolute probability"
        return "In between"

    df["cluster"] = df["prob"].apply(assign_cluster)
    df["confidence_level"] = df["prob"].apply(
        lambda x: "High" if x >= 0.7 else ("Medium" if x > 0.3 else "Low")
    )

    def encode_image(path):
        try:
            if not os.path.isfile(path):
                return ""
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except:
            return ""

    df["uri"] = df["filename"].apply(
        lambda fn: encode_image(os.path.join(image_folder, fn))
    )

    # Filter out images that couldn't be loaded, but preserve original order and indices
    df = df[df["uri"] != ""].copy()

    # Add a unique plot_index that corresponds to the scatter plot points
    df["plot_index"] = range(len(df))

    return df
