# ClusterPhotos

Interesting way with feedback loop of clustering your holiday pics

Make sure to install the local modified dash-cytoscape by either running `pip install -r requirements.txt` or running `pip install -e ./dash-cytoscape`

# Text Clustering 

# Action 40 Classes Visualization

This project provides a visualization interface for action classification data from the Stanford 40 Actions dataset, processed and clustered using GPT-4-based methods.

## ⚠️ Environment Requirement

**Important:** This project only works on **Snellius** (SURF's HPC cluster). Ensure you are on Snellius before proceeding.

## 🧪 Running the Visualization

1. Log into Snellius.  
2. Navigate to your experiment directory:  
   `cd /home/scur0274/Wouter_repo/ClusterPhotos/Text_clustering/ICTC/data/stanford-40-actions/gpt4/action_40_classes/name_your_experiment`  
3. Run the visualization script:  
   `python vis.py`  

This will launch the interface to explore clustering results of the Stanford-40 Actions dataset.

## 📦 Dependencies

Ensure the following Python packages are installed in your Snellius environment:  
matplotlib, seaborn, numpy, pandas, scikit-learn, flask.  
Install them via pip: `pip install matplotlib seaborn numpy pandas scikit-learn flask`

## 📁 Project Structure

    ClusterPhotos/
        Text_clustering/
            ICTC/
                data/
                    stanford-40-actions/
                        gpt4/
                            action_40_classes/
                                name_your_experiment/
                                    vis.py

## 📌 Notes

- Replace `name_your_experiment` with your actual experiment folder name.  
- Ensure any data files required by `vis.py` are present in the experiment directory.

## 📄 License

This repository is intended for academic and research purposes only.
