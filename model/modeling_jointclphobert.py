import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from .module import IntentClassifier, SlotClassifier

class JointCLPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointCLPhoBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained PhoBERT

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_intent_labels,
            self.num_slot_labels,
            self.args.use_intent_context_concat,
            self.args.use_intent_context_attention,
            self.args.max_seq_len,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.margin = args.contrastive_margin if hasattr(args, 'contrastive_margin') else 0.5

    def forward(
        self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids,
        positive_input_ids=None, positive_attention_mask=None, positive_token_type_ids=None,
        negative_input_ids=None, negative_attention_mask=None, negative_token_type_ids=None
    ):
        # Regular Joint Learning
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS] token representation

        intent_logits = self.intent_classifier(pooled_output)
        tmp_attention_mask = attention_mask if self.args.use_attention_mask else None

        if self.args.embedding_type == "hard":
            hard_intent_logits = torch.zeros(intent_logits.shape).to(intent_logits.device)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)

        total_loss = 0
        intent_loss, slot_loss, contrastive_loss = None, None, None

        # 1. Intent Loss
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += self.args.intent_loss_weight * intent_loss

        # 2. Slot Loss
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = -self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean")
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)

            total_loss += self.args.slot_loss_weight * slot_loss

        # 3. Contrastive Loss (if enabled)
        if self.args.use_contrastive_learning and positive_input_ids is not None and negative_input_ids is not None:
            # Forward pass for positive and negative samples
            positive_outputs = self.roberta(
                positive_input_ids, attention_mask=positive_attention_mask, token_type_ids=positive_token_type_ids
            )
            negative_outputs = self.roberta(
                negative_input_ids, attention_mask=negative_attention_mask, token_type_ids=negative_token_type_ids
            )

            # Normalize anchor, positive, and negative representations
            anchor_rep = F.normalize(pooled_output, p=2, dim=1)
            positive_rep = F.normalize(positive_outputs[1], p=2, dim=1)
            negative_rep = F.normalize(negative_outputs[1], p=2, dim=1)

            # Calculate contrastive loss (Triplet Margin Loss)
            contrastive_loss_fct = nn.TripletMarginLoss(margin=self.margin, p=2)
            contrastive_loss = contrastive_loss_fct(anchor_rep, positive_rep, negative_rep)

            total_loss += self.args.contrastive_loss_weight * contrastive_loss

        # outputs = ((intent_logits, slot_logits),) + outputs[2:]
        # outputs = (total_loss,) + outputs
        #
        # return outputs  # (loss), logits, (hidden_states), (attentions)

        # Prepare final outputs
        additional_outputs = outputs[2:]  # Hidden states, attentions
        return (
            total_loss, intent_loss, slot_loss, contrastive_loss,
            (intent_logits, slot_logits), *additional_outputs
        )
