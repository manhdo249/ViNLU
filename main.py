import argparse
from ast import parse

from data_loader import load_and_cache_examples, load_and_cache_triplet_examples
from trainer import Trainer
from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    # Choose dataset loader based on contrastive learning flag
    if args.use_contrastive_learning:
        train_dataset = load_and_cache_triplet_examples(args, tokenizer, mode="train")
    else:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

    # Load dev and test datasets (regular for both cases)
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
    if args.do_eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add new argument to specify if contrastive learning is being used
    parser.add_argument(
        "--use_contrastive_learning",
        action="store_true",
        help="Whether to use contrastive learning with triplet input (anchor, positive, negative samples)"
    )
    parser.add_argument("--contrastive_margin", type=float, default=0.3, help="Margin for triplet loss")

    # Other arguments (kept from the original version)
    parser.add_argument("--negative_class_csv", default="negative_class.csv", type=str, help="Negative class for each class")
    parser.add_argument("--num_negative_samples", type=int, default=3, help="Number of negative samples for each class")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./PhoATIS", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--Number_frozen_block", default=6, type=int, help="Number of block that are frozen in phase 2.")
    parser.add_argument("--epoch_phase1_threshold", default=50, type=int, help="Number of epoch for phase 1.")
    parser.add_argument('--task_num', type=int, help='number of tasks')
    parser.add_argument('--use_decompose', type=int, help='use Decompose')
    parser.add_argument('--decompose_name', type=str, help='Decompose Variant')
    parser.add_argument('--use_MOO', type=int, help='use MOO')
    parser.add_argument('--MOO_name', type=str, help='MOO Variant')

    parser.add_argument("--model_type", default="phobert", type=str, help="Model type selected in the list")
    parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help="Specifies a target value that is ignored and does not contribute to the input gradient")
    # parser.add_argument("--intent_loss_coef", type=float, default=0.5, help="Coefficient for the intent loss.")
    # Loss weights
    parser.add_argument("--intent_loss_weight", type=float, default=1.0, help="Weight for the intent loss.")
    parser.add_argument("--slot_loss_weight", type=float, default=1.0, help="Weight for the slot loss.")
    parser.add_argument("--contrastive_loss_weight", type=float, default=1.0, help="Weight for the contrastive loss.")

    parser.add_argument("--token_level", type=str, default="word-level",
                        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]")
    parser.add_argument("--early_stopping", type=int, default=50,
                        help="Number of unincreased validation step to wait for early stopping")
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")

    # CRF and Pretrained options
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
    parser.add_argument("--pretrained_path", default="./viatis_xlmr_crf", type=str, help="The pretrained model path")

    # Slot-intent interaction options
    parser.add_argument("--use_intent_context_concat", action="store_true",
                        help="Whether to feed context information of intent into slots vectors (simple concatenation)")
    parser.add_argument("--use_intent_context_attention", action="store_true",
                        help="Whether to feed context information of intent into slots vectors (dot product attention)")
    parser.add_argument("--attention_embedding_size", type=int, default=200,
                        help="Hidden size of attention output vector")

    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignored when calculating loss)")
    parser.add_argument("--embedding_type", default="soft", type=str,
                        help="Embedding type for intent vector (hard/soft)")
    parser.add_argument("--use_attention_mask", action="store_true", help="Whether to use attention mask")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
