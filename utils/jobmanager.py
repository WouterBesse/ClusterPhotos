import subprocess


class JobManager:
    def __init__(
        self,
        job_file_path="./Text_clustering/ICTC/step2a.job",
        output_dir="outputs_train",
    ):
        self.job_file_path = job_file_path
        self.output_dir = output_dir
        self.current_job_id = None
        self.job_status = "idle"
        self.current_filter = None  # Track current filter

    def get_current_filter(self):
        """Extract current filter from job file"""
        try:
            with open(self.job_file_path, "r") as f:
                content = f.read()
            lines = content.split("\n")
            for line in lines:
                if line.strip().startswith("--filter"):
                    # Extract filter value from the line
                    filter_part = line.split("--filter")[1].strip()
                    # Remove quotes if present
                    if filter_part.startswith('"') and filter_part.endswith('"'):
                        filter_part = filter_part[1:-1]
                    elif filter_part.startswith("'") and filter_part.endswith("'"):
                        filter_part = filter_part[1:-1]
                    return filter_part
            return "No filter set"
        except Exception as e:
            print(f"Error reading current filter: {e}")
            return "Unknown"

    def update_job_file(self, filter_value):
        try:
            with open(self.job_file_path, "r") as f:
                content = f.read()
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("--filter"):
                    lines[i] = f'  --filter "{filter_value}"'
                    break
            else:
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() and not lines[i].strip().startswith("#"):
                        lines.insert(i + 1, f'  --filter "{filter_value}"')
                        break
            with open(self.job_file_path, "w") as f:
                f.write("\n".join(lines))
            self.current_filter = filter_value  # Update tracked filter
            return True
        except Exception as e:
            print(f"Error updating job file: {e}")
            return False

    def submit_job(self, filter_value):
        if not self.update_job_file(filter_value):
            return False, "Failed to update job file"
        result = subprocess.run(
            ["sbatch", self.job_file_path], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "Submitted batch job" in line:
                    self.current_job_id = line.split()[-1]
                    break
            self.job_status = "running"
            return True, f"Job submitted successfully. Job ID: {self.current_job_id}"
        else:
            return False, f"Failed to submit job: {result.stderr}"

    def check_job_status(self):
        if not self.current_job_id:
            return "idle"
        try:
            result = subprocess.run(
                ["squeue", "-j", self.current_job_id], capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return "running"
                else:
                    self.job_status = "completed"
                    return "completed"
            else:
                return "unknown"
        except Exception as e:
            print(f"Error checking job status: {e}")
            return "unknown"
