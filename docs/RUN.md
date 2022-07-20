# Run
Our code is tested on four NVIDIA Titan RTX GPUs.

## Training

You can use the following command for training our best model.

Please change `[IMG_FEATURE_DIR]` to the directory with Flickr30K's `.npz` feature files. `[PRETRAINED_CHECKPOINT_DIR]` is the Oscar+ pretrained checkpoint and `[OUTPUT_DIR]` is the output directory to dump checkpoints and logs after each epoch.

You can change `--gradient_accumulation_steps` and `--per_gpu_train_batch_size` according to the number of GPUs so that the total batch size per gradient update is 128.

Although we set the total number of epochs to 10 (required for learning rate decay), the best model occurs after 4 epochs of training. We select our best model based on sample-level accuracy on the validation set. If you do early-stopping, rename the fourth checkpoint directory `checkpoint-3` to `best-3` to enable evaluation.

```bash
python oscar/run_ve_amr.py -j 4 \
    --img_feature_dir [IMG_FEATURE_DIR] \
    --img_feature_dim 2054 \
    --max_img_seq_length 50 \
    --data_dir ./data \
    --model_name_or_path [PRETRAINED_CHECKPOINT_DIR] \
    --do_lower_case \
    --max_seq_length 165 \
    --per_gpu_eval_batch_size 64 \
    --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --num_train_epochs 10 \
    --output_dir [OUTPUT_DIR] \
    --loss_type xe \
    --save_epoch 1 \
    --seed 88 \
    --evaluate_during_training \
    --logging_steps 4000 \
    --drop_out 0.3 \
    --type_vocab_size 3 \
    --classifier att+region \
    --init_new_vocab \
    --warmup_steps 0 \
    --lr_decay \
    --ke_loss \
    --struc_loss \
    --loss_weights 0.5 1 1 \
    --do_train 
```

## Evaluation

You can use the following command to test a trained model. For our best model (*Ours+CLS*), the KE accuracy is `68.19%`. You can remove the `--prioritize_cls` flag to get results (KE accuracy `68.07%`) for the model without using CLS predictions (*Ours*).

The fields are the same as above, although `[OUTPUT_DIR]` should now contain the checkpoints that has been finetuned on our task.

```bash
python oscar/run_ve_amr.py -j 4 \
    --img_feature_dir [IMG_FEATURE_DIR] \
    --img_feature_dim 2054 \
    --max_img_seq_length 50 \
    --data_dir ./data \
    --model_type bert \
    --model_name_or_path [PRETRAINED_CHECKPOINT_DIR] \
    --do_lower_case \
    --max_seq_length 165 \
    --per_gpu_eval_batch_size 64 \
    --output_dir [OUTPUT_DIR] \
    --do_test \
    --type_vocab_size 3 \
    --classifier att+region \
    --ke_loss \
    --struc_loss \
    --prioritize_cls
```