"""
Run the entire Customer Churn Intelligence pipeline end-to-end.
Usage: python run_pipeline.py
"""

import sys
import os
import io

# Fix Windows Unicode encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 60)
    print("  CUSTOMER CHURN INTELLIGENCE — FULL PIPELINE")
    print("=" * 60)

    steps = [
        ("1/8", "Generating synthetic dataset",    "src.generate_dataset",    "main"),
        ("2/8", "Cleaning data",                   "src.data_cleaning",       "run_pipeline"),
        ("3/8", "Engineering features",             "src.feature_engineering", "run_feature_engineering"),
        ("4/8", "Running exploratory analysis",     "src.eda",                 "run_eda"),
        ("5/8", "Segmenting customers",             "src.segmentation",        "run_segmentation"),
        ("6/8", "Training ML models",               "src.train_model",         "run_training"),
        ("7/8", "Generating SHAP explanations",     "src.explainability",      "run_explainability"),
        ("8/8", "Generating PDF report",            "src.report_generator",    "generate_report"),
    ]

    for step_num, description, module_name, func_name in steps:
        print(f"\n{'='*60}")
        print(f"  [{step_num}] {description}...")
        print(f"{'='*60}")

        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            func()
        except Exception as e:
            print(f"\n  [FAILED] Step {step_num} failed: {e}")
            print(f"  Stopping pipeline.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print("\n  All data, models, charts, and reports have been generated.")
    print("  Launch the dashboard with: streamlit run dashboard/app.py")
    print()


if __name__ == "__main__":
    main()
