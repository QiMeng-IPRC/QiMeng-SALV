# Signal-aware DPO Stage

This document describes the training data pipeline and the training procedure used in the Signal-aware DPO stage.

## Training Data Pipeline

In the Signal-aware DPO stage, we transform the code-completion descriptions in the CodeV seed data into specification-to-RTL descriptions to enrich and augment the training dataset.
The training data pipeline consists of four steps:

1. Roll out code using the SFT model
    ```
    python ../Utils/rollout.py \
        --model_name <MODEL_PATH> \
        --seed_data <SEED_DATA_PATH> \
        --rollout_save_path <ROLLOUT_OUTPUT> \
        --n 5
    ```
2. Run simulations with iverilog
    ```
    python ../Utils/sim.py \
        --seed_data <SEED_DATA_PATH> \
        --rollout_data <ROLLOUT_DATA_PATH> \
        --sim_info_path <SIM_OUTPUT>
    ```
3. Generate AST text using Yosys
    ```
    python ./gen_AST.py \
        --rollout_data <ROLLOUT_DATA_PATH> \
        --sim_info_path <SIM_INFO_PATH> \
        --ast_save_path <AST_OUTPUT>
    ```
4. Generate the final preference training data

    We use parser.py to extract the code segments corresponding to signal bodding and to produce responses in the format:

    `"verilog code <select_token> code segment token ids"`

    The customized LLaMA-Factory framework parses this format and computes the loss only on the selected tokens.

    ```
    python ./gen_sa_dpo_training_data.py \
        --model_name <MODEL_PATH> \
        --seed_data <SEED_DATA_PATH> \
        --sim_info_path <SIM_INFO_PATH> \
        --rollout_data <ROLLOUT_DATA_PATH> \
        --ast_path <AST_PATH> \
        --training_data_save_path <TRAINING_DATA_OUTPUT>
    ```

To run the full pipeline directly:

```
cd SA-DPO
sh get_sa_dpo_training_data.sh
```

## Training Process

We customize LLaMA-Factory to support response parsing and selective loss computation for Signal-aware DPO training.

Run the following commands to train the Signal-aware DPO model using LoRA and merge the final model:

```
cd LLaMA-Factory
FORCE_TORCHRUN=1 llamafactory-cli train ../SA-DPO/train_sa_dpo.yaml
llamafactory-cli export ../SA-DPO/merge_lora.yaml
```