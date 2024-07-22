from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from dataloader import PretrainDataset
import config
from transformers.generation.configuration_utils import GenerationConfig
from model import GPT2
from tokenizer import SPTokenizer,ChatGLMTokenizer

tokenizer=ChatGLMTokenizer("ice_text.model")


my_dataset=PretrainDataset()

vocab_size = config.VOCAB_SIZE
d_model = config.d_model
num_layers = config.num_layers
num_heads = config.num_heads
d_ff = config.d_ff
max_len = config.max_len
dropout = config.dropout_rate

model = GPT2(vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)

generation_config = GenerationConfig()
generation_config.remove_invalid_values = True
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.pad_token_id = tokenizer.pad_token_id
generation_config.decoder_start_token_id = tokenizer.pad_token_id
generation_config.max_new_tokens = 320
generation_config.num_beams = 1         # greedy search
generation_config.do_sample = False     # greedy search



training_args = Seq2SeqTrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.batch_size_per_gpu,
    auto_find_batch_size=True,  # 防止OOM
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    logging_steps=config.logging_steps,
    num_train_epochs=config.epochs,
    optim="adafactor",
    report_to='tensorboard',
    log_level='info',
    save_steps=config.save_steps,
    save_total_limit=3,
    bf16=True,
    logging_first_step=True,
    warmup_steps=config.warmup_steps,
    seed=config.seed,
    generation_config=generation_config,
)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 准备训练数据
train_examples = [
    {"input": "What is the capital of France?", "target": "The capital of France is Paris."},
    {"input": "Who wrote the play Romeo and Juliet?", "target": "William Shakespeare wrote the play Romeo and Juliet."},
    # 更多训练样本...
]

# 将数据转换为模型输入格式
my_dataset = [
    {"input_ids": tokenizer(example["input"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"],
     "label": tokenizer(example["target"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"]}
    for example in train_examples
]

my_dataset = [
    {"input_ids": tokenizer(example["input"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"],
     "label": tokenizer(example["target"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")["input_ids"]}
    for example in train_examples
]

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#collator = DataCollatorForSeq2Seq(tokenizer, max_length=config.max_len)

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset="my_dataset",
        tokenizer=tokenizer,
        #data_collator=collator,
        #callbacks=[empty_cuda_cahce],
    )
trainer.train()