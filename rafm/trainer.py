import copy
import os
import time
import numpy as np
from .utils import EarlyStopping, step_lr
from .modeling_rafm import RAFM
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments, Trainer
from tqdm import tqdm



def rafm_train(args, model:RAFM, data_shards, val_dataset, test_dataset=None,processor=None, collate_fn=None, compute_metrics=None):
    early_stopping = EarlyStopping(patience=500, verbose=True)

    writer = SummaryWriter(os.path.join(args.save_dir, "logs"))
    best_acc = 0.0
    best_f1 = 0.0
    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "downsize"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        num_train_epochs=args.num_local_epochs,
        learning_rate=args.lr,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        label_names=["labels"],
        # load_best_model_at_end=True,
    )
    eval_args = TrainingArguments(
            output_dir=os.path.join(args.save_dir, "global"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy="no",
            save_strategy="no",
            num_train_epochs=args.num_local_epochs,
            learning_rate=args.lr,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            label_names=["labels"],
            # load_best_model_at_end=True,
    )
    for epoch in tqdm(range(args.epochs), desc="Epoch"):


        lr = step_lr(args.lr, epoch, args.step_size, 0.98)
        training_args.learning_rate = lr
        np.random.seed(int(time.time()))  # Set the seed to the current time

        if args.spp:
            model.salient_parameter_prioritization()
        avg_params = 0
        
        #shuffle the data shards
        np.random.shuffle(data_shards)
        #Train each downsized model independently in a sequential manner
        for idx, data_shard in enumerate(tqdm(data_shards, desc="Mini-shard training")):
            if idx == 0:
                ds_model = copy.deepcopy(model.model)
                ds_model_params = model.total_params
            elif idx == 1:
                (
                    ds_model,
                    ds_model_params,
                    arc_config,
                ) = model.smallest_model()
            else:
                (
                    ds_model,
                    ds_model_params,
                    arc_config,
                ) = model.random_resource_aware_model()

            avg_params += ds_model_params


            trainer = Trainer(
                model=ds_model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=data_shard,
                eval_dataset=val_dataset,
                tokenizer=processor,
            )
            train_results = trainer.train()
            
            model.grad_accumulate(ds_model.to("cpu"),alpha = data_shard.num_rows)
            
    

    
        
        avg_params = avg_params / len(data_shards)
        writer.add_scalar(
            "global/params",
            avg_params,
            epoch,
        )

        # Apply the aggregated and normalized gradient to the full-size model
        model.apply_grad()

        if epoch % 25 == 0:
            # Evaluate the model
            
        
        
            print(f"Training finished for epoch {epoch}")
            
            print("*" * 20 + "Evaluating in epoch {}".format(epoch) + "*" * 20)
            # Evaluate the model at the end of each epoch
            trainer = Trainer(
                model=model.model,
                args=eval_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=None,
                eval_dataset=val_dataset,
                tokenizer=processor,
            )
        
            metrics = trainer.evaluate(val_dataset)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            val_accuracy, val_f1_score = metrics["eval_accuracy"], metrics["eval_f1"]

            writer.add_scalar(
                "global/eval_accuracy",
                val_accuracy,
                epoch,
            )
            writer.add_scalar(
                "global/eval_f1",
                val_f1_score,
                epoch,
            )

    
        
            model.model.to("cpu")
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                model.save_ckpt(os.path.join(args.save_dir, "best_model"))
            if val_f1_score > best_f1:
                best_f1 = val_f1_score
            writer.add_scalar(
                "global/best_accuracy",
                best_acc,
                epoch,
            )
            writer.add_scalar(
                "global/best_f1",
                best_f1,
                epoch,
            )

            print(f"Best validation accuracy: {best_acc} \nBest validation f1: {best_f1}")
            print("*" * 20 + "Evaluation finished" + "*" * 20)

            if test_dataset:
                metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)

            early_stopping(val_f1_score)
            
        model.save_ckpt(os.path.join(args.save_dir, "last_model"))
        if early_stopping.has_converged():
            print("Model has converged. Stopping training.")
            break
    return model

