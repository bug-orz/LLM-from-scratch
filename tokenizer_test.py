from tokenizer import SPTokenizer

temp="### 这是一个测试的tokenizer！"
t=SPTokenizer("ice_text.model")

print(t.encode(temp))
print(t.num_tokens)