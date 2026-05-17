from house_price.advanced_modeling import train_optimized_models
from house_price.advanced_preprocessing import build_advanced_feature_matrix
from house_price.config import DEFAULT_DATA_DIR, PROJECT_ROOT
from house_price.data import load_raw_data, remove_known_outliers


def main() -> None:
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)
    train_clean = remove_known_outliers(train)
    features, target, test_ids = build_advanced_feature_matrix(train_clean, test)

    results = train_optimized_models(
        features=features,
        target=target,
        test_ids=test_ids,
        sample_submission=sample_submission,
        output_dir=PROJECT_ROOT / "outputs",
        submissions_dir=PROJECT_ROOT / "submissions",
        reports_dir=PROJECT_ROOT / "reports",
    )

    print("Optimization round finished.")
    print(f"Feature matrix: {features.train.shape[0]} train rows x {features.train.shape[1]} columns")
    print(f"Best blend: {results.best_submission_name} ({results.best_blend_score:.5f} CV RMSE)")
    print(f"Report: {results.report_path}")
    print("Candidate submissions:")
    for name, path in results.submission_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
