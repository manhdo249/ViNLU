import logging
import os
import numpy as np
import torch
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

from Decomposer.Gram_Schmidt import Gram_Schmidt
from Decomposer.SVD import SVD
from MOO.CAGrad import CAGrad
from MOO.PCGrad import PCGrad
from MOO.DB_MTL import DB_MTL

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        if args.pretrained:
            self.model = self.model_class.from_pretrained(
                args.pretrained_path,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.token_level)
            self.model = self.model_class.from_pretrained(
                args.model_name_or_path,
                config=self.config,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )

        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.args.warmup_steps, t_total)

        # logger.info("***** Running training *****")
        # logger.info(f"  Num examples = {len(self.train_dataset)}")
        # logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        # logger.info(f"  Total optimization steps = {t_total}")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        if self.args.use_decompose == 1:
            if self.args.decompose_name  == 'Gram_Schmidt':
                grad_decomposer = Gram_Schmidt(model=self.model, device='cuda', buffer_size=self.args.task_num*3)
            elif self.args.decompose_name == 'SVD':
                grad_decomposer = SVD(model=self.model, device='cuda', buffer_size=self.args.task_num*3)

        if self.args.use_MOO == 1:
            if self.args.MOO_name == 'PCGrad':
                moo_algorithm = PCGrad()
            elif self.args.MOO_name == 'CAGrad':
                moo_algorithm = CAGrad()
            elif self.args.MOO_name == 'DB_MTL':
                moo_algorithm = DB_MTL(self.args.task_num)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for epoch in train_iterator:
            if (epoch <= self.args.epoch_phase1_threshold):
                print("#### Phase 1")
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {num_params}")
            else:
                print("#### Phase 2")
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Number of trainable parameters: {num_params}")

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch: ", epoch)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack batch depending on contrastive learning flag
                if self.args.use_contrastive_learning:
                    inputs = {
                        "input_ids": batch[0],  # Anchor input
                        "attention_mask": batch[3],
                        "token_type_ids": batch[6],
                        "positive_input_ids": batch[1],  # Positive input
                        "positive_attention_mask": batch[4],
                        "negative_input_ids": batch[2],  # Negative input
                        "negative_attention_mask": batch[5],
                        "intent_label_ids": batch[9],  # Intent labels
                        "slot_labels_ids": batch[10],  # Slot labels
                        "positive_token_type_ids": batch[7],
                        "negative_token_type_ids": batch[8],
                    }
                else:
                    inputs = {
                        "input_ids": batch[0],  # Regular input
                        "attention_mask": batch[1],
                        "token_type_ids": batch[2],
                        "intent_label_ids": batch[3],
                        "slot_labels_ids": batch[4],
                    }

                outputs = self.model(**inputs)
                # loss = outputs[0]
                loss, intent_loss, slot_loss, contrastive_loss = outputs[:4]

                if (epoch <= self.args.epoch_phase1_threshold) or (self.args.use_MOO == 0):
                    '''for param in self.model.roberta.parameters():
                        param.requires_grad = True'''
                    for param in self.model.roberta.embeddings.parameters():
                        param.requires_grad = True

                    for layer in self.model.roberta.encoder.layer[:self.args.Number_frozen_block]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                else:
                    '''for param in self.model.roberta_1.parameters():
                        param.requires_grad = False'''
                    for param in self.model.roberta.embeddings.parameters():
                        param.requires_grad = False

                    for layer in self.model.roberta.encoder.layer[:self.args.Number_frozen_block]:
                        for param in layer.parameters():
                            param.requires_grad = False
                    
                    loss_array = [intent_loss, slot_loss, contrastive_loss]
                    #grad_array = [grad_decomposer._get_total_grad(loss_) for loss_ in loss_array]
                    grad_array = []
                    self.model.zero_grad()
                    grad_array.append(grad_decomposer._get_total_grad(intent_loss))
                    self.model.zero_grad()
                    grad_array.append(grad_decomposer._get_total_grad(slot_loss))
                    self.model.zero_grad()
                    grad_array.append(grad_decomposer._get_total_grad(contrastive_loss))
                    '''print(f"intent gradient shape = {grad_array[0].shape}")
                    print(f"slot gradient shape = {grad_array[1].shape}")
                    print(f"contrastive gradient shape = {grad_array[2].shape}")'''
                    max_size = max(grad.shape[0] for grad in grad_array)
                    grad_array = [torch.cat([grad, grad.new_zeros(max_size - grad.shape[0])]) for grad in grad_array]

                    #total_grad = torch.cat(grad_array)
                    #grad_decomposer.remove_grad_buffer()
                    #grad_decomposer.update_grad_buffer(total_grad)
                    #components = grad_decomposer.decompose_grad(total_grad)

                    adjusted_grad, alpha = moo_algorithm.apply(grad_array)
                    grad_pointer = 0
                    for p in self.model.parameters():
                        if p.requires_grad:
                            num_params = p.numel()
                            grad_slice = adjusted_grad[grad_pointer:grad_pointer + num_params]
                            p.grad = grad_slice.view_as(p).clone()
                            grad_pointer += num_params

                #if self.args.gradient_accumulation_steps > 1:
                #    loss = loss / self.args.gradient_accumulation_steps

                #loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if global_step % self.args.logging_steps == 0:
                        print("\nTuning metrics:", self.args.tuning_metric)
                        print("Loss/train:", tr_loss / global_step)
                        print("Loss Intent/train:", intent_loss)
                        print("Loss Slot/train:", slot_loss)
                        print("Loss Contrastive/train:", contrastive_loss)
                        results = self.evaluate("dev")

                        writer.add_scalar("Loss/validation", results["loss"], epoch)
                        writer.add_scalar("Loss Intent/validation", results["intent_loss"], epoch)
                        writer.add_scalar("Loss Slot/validation", results["slot_loss"], epoch)
                        # writer.add_scalar("Loss Contrastive/validation", results["contrastive_loss"], epoch)
                        writer.add_scalar("Intent Acc/validation", results["intent_acc"], epoch)
                        writer.add_scalar("Slot F1/validation", results["slot_f1"], epoch)
                        writer.add_scalar("Mean Intent Slot/validation", results["mean_intent_slot"], epoch)
                        writer.add_scalar("Sentence Acc/validation", results["semantic_frame_acc"], epoch)

                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            logger.info("Early stopping")
                            break

                if 0 < self.args.max_steps <= global_step:
                    epoch_iterator.close()
                    break

            if early_stopping.early_stop:
                train_iterator.close()
                break
            writer.add_scalar("Loss/train", tr_loss / global_step, epoch)

        return global_step, tr_loss / global_step

    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        dataset = self.dev_dataset if mode == "dev" else self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info(f"***** Running evaluation on {mode} dataset *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        eval_loss, intent_loss_sum, slot_loss_sum = 0.0, 0.0, 0.0
        nb_eval_steps = 0
        intent_preds, slot_preds = None, None
        out_intent_label_ids, out_slot_labels_ids = None, None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "intent_label_ids": batch[3],
                    "slot_labels_ids": batch[4],
                }
                outputs = self.model(**inputs)
                # tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                tmp_eval_loss, intent_loss, slot_loss, contrastive_loss, (intent_logits, slot_logits) = outputs[:5]

                # eval_loss += tmp_eval_loss.mean().item()
                # Accumulate losses
                eval_loss += tmp_eval_loss.mean().item()
                intent_loss_sum += intent_loss.mean().item() if intent_loss is not None else 0.0
                slot_loss_sum += slot_loss.mean().item() if slot_loss is not None else 0.0
                # contrastive_loss_sum += contrastive_loss.mean().item() if contrastive_loss is not None else 0.0

                nb_eval_steps += 1

                # Collect predictions
                if intent_preds is None:
                    intent_preds = intent_logits.detach().cpu().numpy()
                    out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
                else:
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                    out_intent_label_ids = np.append(out_intent_label_ids, inputs["intent_label_ids"].detach().cpu().numpy(), axis=0)

                if slot_preds is None:
                    slot_preds = np.array(self.model.crf.decode(slot_logits)) if self.args.use_crf else slot_logits.detach().cpu().numpy()
                    out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
                else:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                    out_slot_labels_ids = np.append(
                        out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0
                    )

        # eval_loss /= nb_eval_steps
        # results = {"loss": eval_loss}

        # Compute average losses
        avg_eval_loss = eval_loss / nb_eval_steps
        avg_intent_loss = intent_loss_sum / nb_eval_steps
        avg_slot_loss = slot_loss_sum / nb_eval_steps
        # avg_contrastive_loss = contrastive_loss_sum / nb_eval_steps

        results = {
            "loss": avg_eval_loss,
            "intent_loss": avg_intent_loss,
            "slot_loss": avg_slot_loss,
            # "contrastive_loss": avg_contrastive_loss,
        }


        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # print(len(intent_preds), len(out_intent_label_ids), len(slot_preds_list), len(out_slot_label_list))

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        # logger.info("***** Eval results *****")
        # for key, value in results.items():
        #     logger.info(f"  {key} = {value}")
        #
        # return results

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                args=self.args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")