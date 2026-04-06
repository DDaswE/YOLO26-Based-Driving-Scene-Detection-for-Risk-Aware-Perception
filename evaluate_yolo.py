from pathlib import Path

from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "newdata_yolo26_960run_100_32" / "weights" / "best.pt"
DATA_YAML_PATH = BASE_DIR / "dataset.yaml"
OUTPUT_DIR = BASE_DIR / "test_result"
# ---------------------

def main():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        # Load the pre-trained OBB model
        model = YOLO(str(MODEL_PATH))

        print(f"Starting evaluation on data: {DATA_YAML_PATH}...")
        
        # Run validation
        # split='val' is default, but explicit here for clarity
        # save_json=True allows you to parse results later if needed
        metrics = model.val(
            data=str(DATA_YAML_PATH),
            split='test',
            device='mps',  # Use 'cpu' if you don't have a compatible GPU
            workers=2,        # Safe number for clusters
            amp=True,         # Enabled for H100
            project=str(BASE_DIR),
            name='test_result',
            exist_ok=True,
        )

        # The model.val() method automatically prints the table you see during training.
        # However, we can also print specific metrics programmatically:
        
        print("\n" + "="*40)
        print("SPECIFIC METRICS SUMMARY")
        print("="*40)
        
        # Note: metrics.box maps to standard detection, but for OBB models,
        # the library usually populates these attributes with the OBB results.
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall:    {metrics.box.mr:.4f}")
        
        print("\nEvaluation complete.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("Ensure your local paths and dataset.yaml file are correct.")

if __name__ == "__main__":
    main()
