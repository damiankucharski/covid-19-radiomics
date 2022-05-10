def log_metadata(run, config, args, features, features_train, features_test, suffix):
    run['config'] = config._asdict()
    run['suffix'] = suffix
    run['meta/features_source'] = args.features
    run['meta/metadata_source'] = args.metadata
    run['meta/features_shape'] = features.shape
    run['meta/features_train_shape'] = features_train.shape
    run['meta/features_test_shape'] = features_test.shape