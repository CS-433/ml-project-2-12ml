import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, AdamW, get_linear_schedule_with_warmup, pipeline
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

"""

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))
"""


# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
        dataset, model, tokenizer,
        batch_size=16, epochs=5, lr=2e-5,
        max_seq_len=400, warmup_steps=200,
        gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
        test_mode=False
):
    acc_steps = 100
    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
            loss = outputs[0]

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
    return model


# Function to generate multiple sentences. Test data should be a list of strings
def text_generation(test_data):
    generated_text = []
    for i in range(len(test_data)):
        generator = pipeline('text-generation', model=trainedmodel, tokenizer=tokenizer)
        x = generator(test_data[i], max_length=30, num_return_sequences=1)
        generated_text.append(x)
    return generated_text







#TODO adapt into jsonlines
#obtain training data and format it into an array of strings
with open('C:/Users/Antoine/Desktop/UNI/Master/Sem2/ML/MLP2/e-CARE-main/dataset/Explanation_Generation/train.jsonl',
          'r') as json_file:
    json_list = list(json_file)
json_list = [x.replace('"', '').replace(",", '').replace("{", '').replace("}", '').split(' ', 2)[2] for x in json_list]
#printing first string for format checking
print(json_list[0])

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
dataset = []
for elem in json_list:
    dataset.append(torch.tensor(tokenizer.encode(elem)))

trainedmodel = train(dataset, model, tokenizer)

#obtaining test set

with open('C:/Users/Antoine/Desktop/UNI/Master/Sem2/ML/MLP2/e-CARE-main/dataset/Explanation_Generation/dev.jsonl',
          'r') as json_file:
    json_testlist = list(json_file)

json_testlist = [x.replace('"', '').replace(",", '').replace("{", '').replace("}", '').split(' ', 2)[2] for x in json_testlist]
print(json_testlist[0])


#TODO prim generations for comparisons
generated_text = text_generation(json_testlist)
print(generated_text[0])