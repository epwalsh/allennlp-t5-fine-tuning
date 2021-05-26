// =================== Configurable Settings ======================
local debug = true;
// ================================================================

// ---------------- !! Don't edit below here !! -------------------

local model_name = if debug then "t5-small" else "t5-11b";

local data_base_url = "https://storage.googleapis.com/allennlp-public-data/cnndm-combined-data-2020.07.13.tar.gz";
local train_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_train.txt";
local dev_data = data_base_url + "!cnndm-combined-data-2020.07.13/url_lists/all_val.txt";

local dataset_reader = {
    "type": "cnn_dm",
    "source_tokenizer": {
        "type": "pretrained_transformer",
        "model_name": model_name,
    },
    "source_token_indexers": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": model_name,
            "namespace": "tokens",
        }
    },
    "source_max_tokens": 512,
    "target_max_tokens": 54,
    "source_prefix": "summarize: ",
};

{
    "train_data_path": train_data,
    "validation_data_path": dev_data,
    "dataset_reader": dataset_reader + {
        [if debug then "max_instances"]: 64,
    },
    "validation_dataset_reader": dataset_reader + {
        [if debug then "max_instances"]: 32,
    },
    "model": {
        "type": "t5",
        "model_name": model_name,
        // We get the big weights from a beaker dataset.
        [if !debug then "weights_path"]: "/data/t5-11b-weights/t5-11b.bin",
        "beam_size": 3,
        "max_decoding_steps": if debug then 5 else 50,
    },
    // TODO: configure
    "data_loader": {
        "batch_size": 4,
        "shuffle": true,
    },
    "vocabulary": {
        "type": "empty",
    },
    "trainer": {
        "checkpointer": null,
        "num_epochs": 3,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 3e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "correct_bias": true,
        },
        "learning_rate_scheduler": {
            "type": "polynomial_decay",
        },
        "grad_norm": 1.0,
        [if !debug then "callbacks"]: [
            {
                "type": "wandb",
                "project": "allennlp-t5",
                "entity": "allenai-team1",
                "watch_model": false,
                "summary_interval": 1,
                "should_log_parameter_statistics": false,
                "should_log_learning_rate": false,
            },
        ],
    },
    "distributed": {
        "cuda_devices": if debug then [0, 1] else [0, 1, 2, 3, 4, 5, 6, 7],
        "ddp_wrapper": {
            "type": "fairscale_fsdp",
        },
    },
}
