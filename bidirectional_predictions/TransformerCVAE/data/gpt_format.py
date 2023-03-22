import openai
import json, os

openai.api_key = open("creds/OPENAI_API_KEY", "r").read()
num_stories = 200


def load_plots(num_stories: int):
    with open("bidirectional_predictions/TransformerCVAE/data/wikiPlots/titles", "r") as file:
        titles = file.read().split("\n")[:num_stories]

    with open("bidirectional_predictions/TransformerCVAE/data/wikiPlots/plots_paragraph", "r") as file:
        stories = []
        for i in range(num_stories):
            story = ""
            line = file.readline()
            while line[:5] != "<EOS>":
                story += line
                line = file.readline()
            stories.append(story)
            story = ""
    return titles, stories


def prompt_to_story(titles: list[str], stories: list[str]):
    examples = []

    for i in range(len(titles)):
        examples.append({"prompt": titles[i], "completion": f" {stories[i]}"})

    return examples


def sentence_to_sentence(titles: list[str], stories: list[str], prompt_len=1, predict_len=1):
    examples = []

    for story in stories:
        story = story.split(".")
        if len(story[-1]) < 3:
            story.pop()
        while len(story) > prompt_len+predict_len:
            prompt = ""
            size = 0
            while size < prompt_len:
                prompt += story.pop(0)+"."
                size += 1

            predict = ""
            size = 0
            while size < predict_len:
                predict += story.pop(0)+"."
                size += 1

            examples.append({"prompt": prompt, "completion": f" {predict}"})
            
    return examples


def save_examples(examples: list[dict[str, str]], save_file: str):
    with open(save_file, "w") as file:
        for example in examples:

            file.write(json.dumps(example)+"\n")


def upload_file(filename: str, friendly_name: str=None):
    return openai.File.create(file=open(filename, "rb"), purpose='fine-tune', user_provided_filename=friendly_name)


def create_fine_tune(training_file: str, base_model: str):
    return openai.FineTune.create(training_file=training_file, model=base_model)


# titles, stories = load_plots(num_stories)

# print(upload_file("bidirectional_predictions/TransformerCVAE/data/gpt_data.jsonl"))
# ft-DbQ3URrMSQNxdAFODc3u7D4p
# file-8rrm3nBuF8mcfYFUQfKF6Q1b
# ft-bo1UZxw9qReOXWU8N1iUAWpA
# print(create_fine_tune("file-8rrm3nBuF8mcfYFUQfKF6Q1b", "davinci"))
# print(openai.FineTune.list())
# print(openai.FineTune.retrieve(id="ft-DbQ3URrMSQNxdAFODc3u7D4p"))
# examples = sentence_to_sentence(titles, stories, 4, 2)
# examples = prompt_to_story(titles, stories)
# save_examples(examples, "bidirectional_predictions/TransformerCVAE/data/gpt_data.jsonl")