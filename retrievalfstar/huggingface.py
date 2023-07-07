# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small")
model = AutoModel.from_pretrained("kaiyuy/leandojo-lean3-retriever-byt5-small")


print(tokenizer)
print(model)
