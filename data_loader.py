import copy
import json
import logging
import os
import random
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = "seq.in"
        self.intent_label_file = "label"
        self.slot_labels_file = "seq.out"

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = (
                self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            )
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(
                    self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK")
                )

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
            set_type=mode,
        )


processors = {"syllable-level": JointProcessor, "word-level": JointProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
    pad_token_label_id=-100,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len
        )
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_seq_len
        )
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len
        )

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=slot_labels_ids,
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.token_level](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode, args.token_level, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_slot_labels_ids
    )
    return dataset


class TripletInputExample(object):
    """
    A single training example for contrastive learning using triplet inputs.

    Args:
        guid: Unique id for the example.
        anchor_words: list. The words of the anchor sequence.
        positive_words: list. The words of the positive sequence (same intent).
        negative_words: list. The words of the negative sequence (different intent).
        intent_label: (Optional) string. The intent label of the anchor example.
        slot_labels: (Optional) list. The slot labels of the anchor example.
    """

    def __init__(self, guid, anchor_words, positive_words, negative_words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.anchor_words = anchor_words
        self.positive_words = positive_words
        self.negative_words = negative_words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class TripletInputFeatures(object):
    """
    A single set of triplet features for contrastive learning.
    """

    def __init__(self,
                 anchor_input_ids, anchor_attention_mask, anchor_token_type_ids,
                 positive_input_ids, positive_attention_mask, positive_token_type_ids,
                 negative_input_ids, negative_attention_mask, negative_token_type_ids,
                 intent_label_id, slot_labels_ids):
        self.anchor_input_ids = anchor_input_ids
        self.anchor_attention_mask = anchor_attention_mask
        self.anchor_token_type_ids = anchor_token_type_ids

        self.positive_input_ids = positive_input_ids
        self.positive_attention_mask = positive_attention_mask
        self.positive_token_type_ids = positive_token_type_ids

        self.negative_input_ids = negative_input_ids
        self.negative_attention_mask = negative_attention_mask
        self.negative_token_type_ids = negative_token_type_ids

        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessorForTriplet(JointProcessor):
    """Processor for creating triplet examples for contrastive learning."""
    def __init__(self, args):
        super().__init__(args)
        # Load the CSV file mapping anchor classes to additional negative classes
        self.negative_class_map = self._load_negative_class_map(args.negative_class_csv)

    def _load_negative_class_map(self, csv_path):
        """Load CSV containing anchor-to-negative class mappings."""
        data_path = os.path.join(self.args.data_dir, self.args.token_level)

        df = pd.read_csv(os.path.join(data_path, csv_path))
        negative_class_map = {}
        for _, row in df.iterrows():
            anchor_class = row['anchor_class']
            negative_class = row['negative_class']
            if anchor_class not in negative_class_map:
                negative_class_map[anchor_class] = []
            negative_class_map[anchor_class].append(negative_class)
        return negative_class_map

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info("LOOKING AT {}".format(data_path))

        # Read texts, intents, and slots from the files
        texts = self._read_file(os.path.join(data_path, self.input_text_file))
        intents = self._read_file(os.path.join(data_path, self.intent_label_file))
        slots = self._read_file(os.path.join(data_path, self.slot_labels_file))

        # Group examples by intent for easy positive/negative selection
        intent_groups = {}
        for text, intent, slot in zip(texts, intents, slots):
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            if intent_label not in intent_groups:
                intent_groups[intent_label] = []
            intent_groups[intent_label].append((text, slot))

        examples = []
        for i, (anchor_text, anchor_intent, anchor_slot) in enumerate(zip(texts, intents, slots)):
            guid = f"{mode}-{i}"
            anchor_words = anchor_text.split()
            anchor_intent_label = self.intent_labels.index(anchor_intent) if anchor_intent in self.intent_labels else self.intent_labels.index("UNK")
            anchor_slot_labels = [self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK") for s in anchor_slot.split()]

            # Step 1: Select 1 negative sample from each different intent class
            negative_samples = self._get_negatives_from_other_classes(anchor_intent_label, intent_groups)

            # random select 3 negative samples
            negative_samples = random.sample(negative_samples, self.args.num_negative_samples-2)

            # Step 2: Add 2 more negatives from specific negative classes (from CSV)
            additional_negatives = self._get_negatives_from_csv(anchor_intent_label, intent_groups, num_negatives=2)
            negative_samples.extend(additional_negatives)

            # Step 3: Balance positive samples to match the number of negatives
            positive_samples = self._get_balanced_positive_samples(anchor_intent_label, intent_groups,
                                                                   len(negative_samples))

            # Step 4: Create triplets with each positive-negative pair
            for pos, neg in zip(positive_samples, negative_samples):
                positive_words = pos[0].split()
                negative_words = neg[0].split()

                examples.append(TripletInputExample(
                    guid=guid,
                    anchor_words=anchor_words,
                    positive_words=positive_words,
                    negative_words=negative_words,
                    intent_label=anchor_intent_label,
                    slot_labels=anchor_slot_labels
                ))

        return examples

    def _get_negatives_from_other_classes(self, intent_label, intent_groups):
        """Get one negative sample from each different intent class."""
        negative_samples = []
        other_intents = [label for label in intent_groups if label != intent_label]
        for neg_intent_label in other_intents:
            negative_samples.append(random.choice(intent_groups[neg_intent_label]))
        return negative_samples

    def _get_negatives_from_csv(self, intent_label, intent_groups, num_negatives=2):
        """Get additional negatives from classes specified in the CSV mapping."""
        additional_negatives = []
        mapped_neg_classes = self.negative_class_map.get(intent_label, [])
        for neg_class in mapped_neg_classes:
            if neg_class in intent_groups:
                additional_negatives.extend(random.sample(intent_groups[neg_class], min(num_negatives, len(intent_groups[neg_class]))))
        return additional_negatives[:num_negatives]  # Limit to `num_negatives`

    def _get_balanced_positive_samples(self, intent_label, intent_groups, num_samples):
        """Get a balanced number of positive samples to match the number of negatives."""
        positive_samples = intent_groups[intent_label]
        if len(positive_samples) >= num_samples:
            return random.sample(positive_samples, num_samples)
        else:
            return positive_samples * (num_samples // len(positive_samples)) + random.sample(positive_samples, num_samples % len(positive_samples))



def convert_triplet_examples_to_features(
        examples,
        max_seq_len,
        tokenizer,
        pad_token_label_id=-100,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        def tokenize_and_pad(words, slot_labels):
            tokens = []
            slot_labels_ids = []
            for word, slot_label in zip(words, slot_labels):
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # Handling bad-encoded words
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Truncate if tokens are longer than the max_seq_len minus [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[: (max_seq_len - special_tokens_count)]
                slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            slot_labels_ids += [pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_labels_ids = [pad_token_label_id] + slot_labels_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

            # Convert tokens to input IDs
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Padding up to the max sequence length
            padding_length = max_seq_len - len(input_ids)
            input_ids += [pad_token_id] * padding_length
            attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
            token_type_ids += [pad_token_segment_id] * padding_length
            slot_labels_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
                len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
                len(token_type_ids), max_seq_len)
            assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
                len(slot_labels_ids), max_seq_len)

            return input_ids, attention_mask, token_type_ids, slot_labels_ids

        # Process anchor, positive, and negative examples
        anchor_input_ids, anchor_attention_mask, anchor_token_type_ids, anchor_slot_labels_ids = tokenize_and_pad(
            example.anchor_words, example.slot_labels)
        positive_input_ids, positive_attention_mask, positive_token_type_ids, _ = tokenize_and_pad(
            example.positive_words, example.slot_labels)
        negative_input_ids, negative_attention_mask, negative_token_type_ids, _ = tokenize_and_pad(
            example.negative_words, example.slot_labels)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"anchor input_ids: {' '.join([str(x) for x in anchor_input_ids])}")
            logger.info(f"positive input_ids: {' '.join([str(x) for x in positive_input_ids])}")
            logger.info(f"negative input_ids: {' '.join([str(x) for x in negative_input_ids])}")
            logger.info(f"intent_label: {example.intent_label} (id = {intent_label_id})")
            logger.info(f"slot_labels: {' '.join([str(x) for x in anchor_slot_labels_ids])}")

        # Append the processed input features to the feature list
        features.append(
            TripletInputFeatures(
                anchor_input_ids=anchor_input_ids,
                anchor_attention_mask=anchor_attention_mask,
                anchor_token_type_ids=anchor_token_type_ids,
                positive_input_ids=positive_input_ids,
                positive_attention_mask=positive_attention_mask,
                positive_token_type_ids=positive_token_type_ids,
                negative_input_ids=negative_input_ids,
                negative_attention_mask=negative_attention_mask,
                negative_token_type_ids=negative_token_type_ids,
                intent_label_id=intent_label_id,
                slot_labels_ids=anchor_slot_labels_ids  # Only anchor's slot labels are used
            )
        )

    return features


def load_and_cache_triplet_examples(args, tokenizer, mode):
    processor = JointProcessorForTriplet(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_triplet_{}_{}_{}_{}".format(
            mode, args.token_level, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading triplet features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating triplet features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("Only train, dev, and test modes are available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_triplet_examples_to_features(
            examples, args.max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id
        )
        logger.info("Saving triplet features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset for triplet input
    all_anchor_input_ids = torch.tensor([f.anchor_input_ids for f in features], dtype=torch.long)
    all_positive_input_ids = torch.tensor([f.positive_input_ids for f in features], dtype=torch.long)
    all_negative_input_ids = torch.tensor([f.negative_input_ids for f in features], dtype=torch.long)

    all_anchor_attention_mask = torch.tensor([f.anchor_attention_mask for f in features], dtype=torch.long)
    all_positive_attention_mask = torch.tensor([f.positive_attention_mask for f in features], dtype=torch.long)
    all_negative_attention_mask = torch.tensor([f.negative_attention_mask for f in features], dtype=torch.long)

    all_anchor_token_type_ids = torch.tensor([f.anchor_token_type_ids for f in features], dtype=torch.long)
    all_positive_token_type_ids = torch.tensor([f.positive_token_type_ids for f in features], dtype=torch.long)
    all_negative_token_type_ids = torch.tensor([f.negative_token_type_ids for f in features], dtype=torch.long)

    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_anchor_input_ids, all_positive_input_ids, all_negative_input_ids,
        all_anchor_attention_mask, all_positive_attention_mask, all_negative_attention_mask,
        all_anchor_token_type_ids, all_positive_token_type_ids, all_negative_token_type_ids,
        all_intent_label_ids, all_slot_labels_ids
    )

    return dataset
