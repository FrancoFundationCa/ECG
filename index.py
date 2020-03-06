    ds.Pipeline()

      .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)

      .init_variable("loss_history", init_on_each_run=list)

      .load(components=["signal", "meta"], fmt="wfdb")

      .load(components="target", fmt="csv", src=LABELS_PATH)

      .drop_labels(["~"])

      .rename_labels({"N": "NO", "O": "NO"})

      .flip_signals()

      .random_resample_signals("normal", loc=300, scale=10)

      .random_split_signals(2048, {"A": 9, "NO": 3})

      .binarize_labels()

      .train_model("dirichlet", make_data=concatenate_ecg_batch, fetches="loss", save_to=V("loss_history"), mode="a")

      .run(batch_size=100, shuffle=True, drop_last=True, n_epochs=50)

)
