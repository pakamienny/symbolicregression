# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from symbolicregression.utils import bool_flag


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="SR", add_help=False)

    # main parameters
    parser.add_argument(
        "--dump_path", type=str, default="", help="Experiment dump path"
    )
    parser.add_argument(
        "--eval_dump_path", type=str, default=None, help="Evaluation dump path"
    )
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument(
        "--print_freq", type=int, default=100, help="Print every n steps"
    )
    parser.add_argument(
        "--save_periodic",
        type=int,
        default=25,
        help="Save the model periodically (0 to disable)",
    )
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")

    # float16 / AMP API
    parser.add_argument(
        "--fp16", type=bool_flag, default=False, help="Run model with float16"
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=-1,
        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.",
    )
    # model parameters
    parser.add_argument(
        "--embedder_type",
        type=str,
        default="conv",
        help="[TNet, LinearPoint, Flat, AttentionPoint] How to pre-process sequences before passing to a transformer.",
    )
    parser.add_argument(
        "--emb_conv_kernel", type=int, default=12, help="Embedder conv size"
    )
    parser.add_argument(
        "--emb_conv_stride", type=int, default=12, help="Embedder conv stride"
    )
    parser.add_argument(
        "--emb_emb_dim", type=int, default=64, help="Embedder embedding layer size"
    )
    parser.add_argument(
        "--enc_emb_dim", type=int, default=512, help="Encoder embedding layer size"
    )
    parser.add_argument(
        "--dec_emb_dim", type=int, default=512, help="Decoder embedding layer size"
    )
    parser.add_argument(
        "--n_emb_layers", type=int, default=1, help="Number of layers in the embedder",
    )
    parser.add_argument(
        "--n_enc_layers",
        type=int,
        default=4,
        help="Number of Transformer layers in the encoder",
    )
    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=8,
        help="Number of Transformer layers in the decoder",
    )
    parser.add_argument(
        "--n_enc_heads",
        type=int,
        default=16,
        help="Number of Transformer encoder heads",
    )
    parser.add_argument(
        "--n_dec_heads",
        type=int,
        default=16,
        help="Number of Transformer decoder heads",
    )
    parser.add_argument(
        "--emb_expansion_factor",
        type=int,
        default=1,
        help="Expansion factor for embedder",
    )
    parser.add_argument(
        "--n_enc_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer encoder",
    )
    parser.add_argument(
        "--n_dec_hidden_layers",
        type=int,
        default=1,
        help="Number of FFN layers in Transformer decoder",
    )

    parser.add_argument(
        "--norm_attention",
        type=bool_flag,
        default=False,
        help="Normalize attention and train temperaturee in Transformer",
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout")
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0,
        help="Dropout in the attention layer",
    )
    parser.add_argument(
        "--share_inout_emb",
        type=bool_flag,
        default=True,
        help="Share input and output embeddings",
    )
    parser.add_argument(
        "--enc_positional_embeddings",
        type=str,
        default=None,
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )
    parser.add_argument(
        "--dec_positional_embeddings",
        type=str,
        default="learnable",
        help="Use none/learnable/sinusoidal/alibi embeddings",
    )

    parser.add_argument(
        "--env_base_seed",
        type=int,
        default=0,
        help="Base seed for environments (-1 to use timestamp seed)",
    )
    parser.add_argument(
        "--test_env_seed", type=int, default=1, help="Test seed for environments"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Number of sentences per batch"
    )
    parser.add_argument(
        "--batch_size_eval",
        type=int,
        default=1,
        help="Number of sentences per batch during evaluation (if None, set to 1.5*batch_size)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_inverse_sqrt,warmup_updates=10000",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=0.5,
        help="Clip gradients norm (0 to disable)",
    )
    parser.add_argument(
        "--n_steps_per_epoch", type=int, default=3000, help="Number of steps per epoch",
    )
    parser.add_argument(
        "--max_epoch", type=int, default=100000, help="Number of epochs"
    )
    parser.add_argument(
        "--stopping_criterion",
        type=str,
        default="",
        help="Stopping criterion, and number of non-increase before stopping the experiment",
    )

    parser.add_argument(
        "--accumulate_gradients",
        type=int,
        default=1,
        help="Accumulate model gradients over N iterations (N times larger batch sizes)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )

    parser.add_argument(
        "--train_noise_gamma",
        type=float,
        default=0.0,
        help="Should we train with additional output noise",
    )

    parser.add_argument(
        "--ablation_to_keep",
        type=str,
        default=None,
        help="which ablation should we do",
    )

    parser.add_argument(
        "--max_input_points",
        type=int,
        default=200,
        help="split into chunks of size max_input_points at eval",
    )
    parser.add_argument(
        "--n_trees_to_refine", type=int, default=10, help="refine top n trees"
    )

    # export data / reload it
    parser.add_argument(
        "--export_data",
        type=bool_flag,
        default=False,
        help="Export data and disable training.",
    )
    parser.add_argument(
        "--reload_data",
        type=str,
        default="",
        help="folder",
    )
    parser.add_argument(
        "--reload_size",
        type=int,
        default=-1,
        help="Reloaded training set size (-1 for everything)",
    )
    parser.add_argument(
        "--batch_load",
        type=bool_flag,
        default=False,
        help="Load training set by batches (of size reload_size).",
    )

    # environment parameters
    parser.add_argument(
        "--env_name", type=str, default="sr", help="Environment name"
    )
    # tasks
    parser.add_argument("--tasks", type=str, default="sr", help="Tasks")

    # beam search configuration
    parser.add_argument(
        "--beam_eval",
        type=bool_flag,
        default=True,
        help="Evaluate with beam search decoding.",
    )
    parser.add_argument(
        "--max_generated_output_len",
        type=int,
        default=200,
        help="Max generated output length",
    )
    parser.add_argument(
        "--beam_eval_train",
        type=int,
        default=0,
        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size, default = 1 (greedy decoding)",
    )
    parser.add_argument(
        "--beam_type", type=str, default="sampling", help="Beam search or sampling",
    )
    parser.add_argument(
        "--beam_temperature",
        type=int,
        default=0.8,
        help="Beam temperature for sampling",
    )

    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=1,
        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.",
    )
    parser.add_argument(
        "--beam_early_stopping",
        type=bool_flag,
        default=True,
        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.",
    )
    parser.add_argument("--beam_selection_metrics", type=int, default=1)

    parser.add_argument("--max_number_bags", type=int, default=1)

    # reload pretrained model / checkpoint
    parser.add_argument(
        "--reload_model", type=str, default="", help="Reload a pretrained model"
    )
    parser.add_argument(
        "--reload_checkpoint", type=str, default="", help="Reload a checkpoint"
    )



    # evaluation
    parser.add_argument(
        "--validation_metrics",
        type=str,
        default="r2_train_median",
        help="What metrics should we report? accuracy_tolerance/_l1_error/r2/_complexity/_relative_complexity/is_symbolic_solution",
    )

    parser.add_argument(
        "--debug_train_statistics",
        type=bool,
        default=False,
        help="whether we should print infos distributions",
    )

    parser.add_argument(
        "--eval_noise_gamma",
        type=float,
        default=0.0,
        help="Should we evaluate with additional output noise",
    )
    parser.add_argument(
        "--eval_size", type=int, default=100, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_noise_type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Type of noise added at test time",
    )
    parser.add_argument(
        "--eval_noise", type=float, default=0, help="Size of valid and test samples"
    )
    parser.add_argument(
        "--eval_only", type=bool_flag, default=False, help="Only run evaluations"
    )
    parser.add_argument(
        "--eval_from_exp", type=str, default="", help="Path of experiment to use"
    )
    parser.add_argument(
        "--eval_data", type=str, default="", help="Path of data to eval"
    )
    parser.add_argument(
        "--eval_verbose", type=int, default=0, help="Export evaluation details"
    )
    parser.add_argument(
        "--eval_verbose_print",
        type=bool_flag,
        default=False,
        help="Print evaluation details",
    )
    parser.add_argument(
        "--eval_input_length_modulo",
        type=int,
        default=-1,
        help="Compute accuracy for all input lengths modulo X. -1 is equivalent to no ablation",
    )
    parser.add_argument("--eval_on_pmlb", type=bool, default=True)
    parser.add_argument("--eval_in_domain", type=bool, default=True)

    # debug
    parser.add_argument(
        "--debug_slurm",
        type=bool_flag,
        default=False,
        help="Debug multi-GPU / multi-node within a SLURM job",
    )
    parser.add_argument("--debug", help="Enable all debug flags", action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False, help="Run on CPU")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument(
        "--windows",
        type=bool_flag,
        default=False,
        help="Windows version (no multiprocessing for eval)",
    )
    parser.add_argument(
        "--nvidia_apex", type=bool_flag, default=False, help="NVIDIA version of apex"
    )

    parser.add_argument(
        "--queue_strategy",
        type=str,
        default="uniform_sampling",
        help="in [precompute_batches, uniform_sampling, uniform_sampling_replacement]",
    )

    parser.add_argument("--use_emb_positional_embeddings", type=bool, default=False)

    parser.add_argument("--collate_queue_size", type=int, default=2000)
    parser.add_argument(
            "--tokens_per_batch",
            type=int,
            default=10000,
            help="max number of tokens per batch",
        )        
    parser.add_argument(
            "--max_token_len",
            type=int,
            default=0,
            help="max size of tokenized sentences, 0 is no filtering",
        )
    parser.add_argument(
        "--n_min_observations",
        type=int,
        default=20,
        help="max number of observations",
    )
    parser.add_argument(
        "--n_max_observations",
        type=int,
        default=100,
        help="max number of observations",
    )

    parser.add_argument(
        "--unary_ops_str",
        type=str,
        default="",
        help="Unary operators. Empty string for all.",
    )
    parser.add_argument(
        "--binary_ops_str",
        type=str,
        default="",
        help="Binary operators. Empty string for all.",
    )
    parser.add_argument(
        "--leaf_probs_str",
        type=str,
        default="0.5,0.5",
        help="Leaf probabilities (float, variable). Must sum to 1",
    )
    parser.add_argument(
        "--operators_upsample_str",
        type=str,
        default="abs_0.2,inv_0.3,sin_0.5,cos_0.5,exp_0.2,log_0.2,tan_0.2,sqrt_0.3",
        help="OPNAME_UpSampleFactor",
    )


    parser.add_argument(
        "--min_ops",
        type=int,
        default=5,
        help="max number of operators",
    )
    parser.add_argument(
        "--max_ops",
        type=int,
        default=25,
        help="max number of operators",
    )
    parser.add_argument(
        "--min_vars",
        type=int,
        default=1,
        help="min number of variables",
    )
    parser.add_argument(
        "--max_vars",
        type=int,
        default=5,
        help="max number of variables",
    )

    parser.add_argument(
        "--pad_to_max_dim",
        type=bool,
        default=True,
        help="should we pad",
    )
    parser.add_argument(
        "--save_results_every",
        type=int,
        default=1,
        help="max number of variables",
    )
    parser.add_argument(
        "--srbench_path",
        type=str,
        default="/private/home/pakamienny/Research_2/pmlb/filtered_datasets/",
        help="path to srbench datasets",
    )
    return parser

default_params = get_parser().parse_args(args=[])
if __name__ == "__main__":
    print(default_params)