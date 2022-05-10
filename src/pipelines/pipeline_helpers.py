
def run_transformer_pipeline_on_train_and_test(pipeline, features, features_train, features_test, y_train=None):
    train_transformed = pipeline.fit_transform(features_train, y_train)
    test_transformed = pipeline.transform(features_test)

    features = features.loc[:, train_transformed.columns]
    features.loc[train_transformed.index, :] = train_transformed
    features.loc[test_transformed.index, :] = test_transformed

    return features, train_transformed, test_transformed