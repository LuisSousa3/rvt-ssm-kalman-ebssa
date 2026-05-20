"""Fixed settings for the EBSSA/RVT experiment used in this project."""


class ProjectDefaults:
    # Conversion and dataset layout
    limit = None
    stats_only = False
    output_kind = "rvt-raw"

    preview_mat_file = None
    preview_output_path = None
    preview_output_format = "mp4"
    sample_plot_mat_file = None
    preview_frame_duration_us = 10_000
    preview_frame_step_us = 10_000
    preview_n_bins = 10
    preview_fps = 10
    preview_max_frames = None
    preview_start_time_us = None
    preview_end_time_us = None
    preview_draw_labels = True
    preview_label_time_tolerance_us = 100_000
    preview_bbox_size = None
    preview_box_line_width = 1
    preview_scale = 4

    mat_data_dir = "data/ebssa"
    gen1_data_dir = "data/ebssa_gen1"
    rvt_raw_data_dir = "data/ebssa_gen1_raw"
    rvt_preprocessed_data_dir = "data/ebssa_gen1_preprocessed"
    plots_dir = "data/ebssa_plots"

    atis_size = (240, 304)
    davis_size = (240, 304)
    use_mat_sensor_size = False
    bbox_size = 22.0
    class_id = 0
    frame_duration_us = 50_000
    n_bins = 10
    segment_duration_us = 60_000_000
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    split_seed = 42
    include_unlabeled = False
    min_box_diag = None
    min_box_side = None

    # RVT preprocessing and final training recipe
    rvt_dataset_name = "gen1"
    rvt_num_classes = 1
    rvt_class_names = ("object",)
    rvt_preprocess_num_processes = 4
    rvt_preprocess_representation_config = (
        "RVT/scripts/genx/conf_preprocess/representation/stacked_hist.yaml"
    )
    rvt_preprocess_extraction_config = (
        "RVT/scripts/genx/conf_preprocess/extraction/const_duration.yaml"
    )
    rvt_preprocess_filter_config = "RVT/scripts/genx/conf_preprocess/filter_gen1.yaml"

    rvt_dry_run = False
    rvt_python_executable = "/home/lsousa/anaconda3/envs/ebssa_ssm/bin/python"
    rvt_csv_log_dir = "data/rvt_csv_logs"
    rvt_test_name = "modelFinal"
    rvt_output_root = "ebssa_rvt"
    rvt_log_group = "ebssa_gen1_s5_transfer"
    rvt_experiment_config = "base"
    rvt_accelerator = "gpu"
    rvt_gpus = "0"
    rvt_batch_size_train = 2
    rvt_batch_size_eval = 2
    rvt_num_workers_train = 8
    rvt_num_workers_eval = 2
    rvt_max_epochs = 20
    rvt_max_steps = 100_000
    rvt_learning_rate = 1e-4
    rvt_weight_decay = 1e-4
    rvt_precision = 32
    rvt_use_lr_scheduler = True
    rvt_lr_scheduler_pct_start = 0.005
    rvt_lr_scheduler_div_factor = 1.0
    rvt_lr_scheduler_final_div_factor = 2.0
    rvt_val_check_interval = 500
    rvt_check_val_every_n_epoch = None
    rvt_limit_train_batches = 1.0
    rvt_limit_val_batches = 1.0
    rvt_log_every_n_steps = 50
    rvt_enable_visual_logging = False
    rvt_generate_metric_plots = True
    rvt_metric_plots_dir = "outputs/rvt_training_plots"
    rvt_event_representation_name = "stacked_histogram_dt=50_nbins=10"
    rvt_train_sampling = "mixed"
    rvt_eval_sampling = "stream"
    rvt_sequence_length = 21
    rvt_input_channels = 20
    rvt_model_embed_dim = 64
    rvt_model_dim_head = 32
    rvt_model_fpn_depth = 0.67
    rvt_model_partition_split_32 = 1
    rvt_s5_state_dim = None
    rvt_pretrained_checkpoint_path = "data/gen1_base.ckpt"
    rvt_pretrained_reset_classification_head = True
    rvt_extra_train_overrides = ()

    # Checkpoint selection and evaluation
    rvt_checkpoint_dir = None
    rvt_checkpoint_selection_metric = "val_AP"
    rvt_checkpoint_selection_mode = "max"
    rvt_eval_output_dir = None
    rvt_eval_iou_threshold = 0.50
    rvt_eval_confidence_threshold = 0.10
    rvt_eval_max_test_sequences = None

    # memory and seq length studies
    run_rvt_s5_memory_study = False
    run_rvt_sequence_length_study = False
    rvt_study_max_test_sequences = None
    rvt_s5_memory_reset_intervals = (None, 21, 10, 5, 1)
    rvt_sequence_length_study_values = (1, 5, 10, 21)
    rvt_sequence_length_study_reset_to_window = True

    rvt_eval_video_sequence_name = None #("data/ebssa_gen1_preprocessed/test/20170214-21-15_SL8RB_21938_labelled")
    rvt_eval_video_all_test_sequences = True
    rvt_eval_video_max_frames = 600
    rvt_eval_video_fps = 1
    rvt_eval_video_scale = 4
    rvt_eval_video_draw_ground_truth = True

    # Kalman tracking settings used only in the post-processing comparison
    rvt_kalman_association_iou = 0.10
    rvt_kalman_association_center_distance_scale = 2.0
    rvt_kalman_birth_suppression_center_distance_scale = 2.0
    rvt_kalman_max_missed_frames = 20
    rvt_kalman_min_confirmed_hits = 2
    rvt_kalman_tentative_max_missed_frames = 0
    rvt_kalman_process_noise = 20.0
    rvt_kalman_measurement_noise = 8.0
