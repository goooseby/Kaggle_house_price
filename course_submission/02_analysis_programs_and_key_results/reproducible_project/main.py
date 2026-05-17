
from house_price.config import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR
from house_price.data import load_raw_data, remove_known_outliers
from house_price.eda import write_eda_report
from house_price.modeling import train_and_predict
from house_price.preprocessing import build_feature_matrix


def main() -> None:
    train, test, sample_submission = load_raw_data(DEFAULT_DATA_DIR)

    write_eda_report(train, test, DEFAULT_OUTPUT_DIR / "eda_report.md")

    train_clean = remove_known_outliers(train)
    features, target, test_ids = build_feature_matrix(train_clean, test)

    results = train_and_predict(
        features=features,
        target=target,
        test_ids=test_ids,
        sample_submission=sample_submission,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    print("Baseline finished.")
    print(f"Train rows after outlier removal: {len(train_clean)}")
    print(f"Feature matrix: {features.train.shape[0]} train rows x {features.train.shape[1]} columns")
    print(f"Best single model: {results.best_model_name} ({results.best_score:.5f} CV RMSE)")
    print(f"Blend CV RMSE: {results.blend_score:.5f}")
    print(f"Submission: {results.submission_path}")
    print(f"EDA report: {DEFAULT_OUTPUT_DIR / 'eda_report.md'}")


if __name__ == "__main__":
    main()
