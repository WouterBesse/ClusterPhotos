# ClusterPhotos

Interesting way with feedback loop of clustering your holiday pics

Make sure to install the local modified dash-cytoscape by either running `pip install -r requirements.txt` or running `pip install -e ./dash-cytoscape`

# Text Clustering 

# Custom Data Clustering Visualization

This project provides a visualization interface for clustering your own custom dataset using an arbitrary filter criterion.

## ⚠️ Environment Requirement

**Important:** This project only works on **Snellius** (SURF's HPC cluster). Ensure you are logged into Snellius before proceeding.

## 🧪 Running the Visualization

1. Log into Snellius.  
2. Navigate to your experiment directory:  
   `cd /home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/data/stanford-40-actions/gpt4/action_40_classes/name_your_experiment`  
3. Run the visualization script:  
   `python vis.py`  

This will launch the interface allowing you to:
- Apply a custom filter (e.g. by keyword, date range, user ID, etc.)  
- View clusters generated based on that filter

## 📦 Dependencies

All required Python packages are listed in the `requirements.txt` file located in the `Text_clustering` folder. Install them with:  
`pip install -r /home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/requirements.txt`

## 📁 Project Structure

    ClusterPhotos/
        Text_clustering/
            requirements.txt
            ICTC/
                data/
                    stanford-40-actions/      ← you can ignore or repurpose this folder
                        gpt4/
                            action_40_classes/
                                name_your_experiment/
                                    vis.py
                                    your_custom_data.csv
