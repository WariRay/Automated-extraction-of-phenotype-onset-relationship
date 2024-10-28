import yaml
import json
from os import path
from os import environ

OPENAI_ORG = environ["OPENAI_ORG"]
OPENAI_KEY = environ["OPENAI_KEY"]

def parse_prompt_template(file_path, output_file_path):
    with open(file_path, "r") as file:
        yaml_template = yaml.safe_load(file)
    json_template = json.dumps(yaml_template, indent=4)
    with open(output_file_path, "w") as out_file:
        out_file.write(json_template)
    print(f"Prompt template was processed successfully.")
    return json_template

SMALL_DATASET_DIR = path.join("small_dataset")
SMALL_TEXT_DATA = path.join(SMALL_DATASET_DIR, "test_small_text.txt")
SMALL_PHENO_LIST = path.join(SMALL_DATASET_DIR, "phenos.txt")

def generate_data_prompts(prompt_template):
    # Process prompt template
    with open(prompt_template, "r") as file:
        json_template = file.read()

    #  Text data file path
    with open("small_dataset/test_small_text.txt", "r") as file:
        clinical_text = file.read()

    # Phenotypes file path
    with open("small_dataset/phenos.txt", "r") as file:
        phenos_content = file.read()
    phenos_list = phenos_content.split("\n")

    # Onset file path
    with open("prompt_templates/onset_definitions.yml", "r") as file:
        ymal_file = yaml.safe_load(file)
    onset_dict = ymal_file

    # Get all the onsets
    onsets_list = []
    for onset in onset_dict:
        onsets_list.append(onset)

    pheno_onset_pairs = []
    prompts_list = []
    onset_definitition = ""

    for pheno in phenos_list:
        for onset in onsets_list:
            prompt_str = json_template
            onset_definitition = onset_dict[onset]
            prompt_str = prompt_str.replace("{{clinical_text}}", clinical_text)
            prompt_str = prompt_str.replace("{{phenotype}}", pheno)
            prompt_str = prompt_str.replace("{{onset}}", onset)
            prompt_str = prompt_str.replace("{{onset_definition}}", onset_definitition)

            try:
                prompt = json.loads(prompt_str, strict=False)
                prompts_list.append(prompt)
                pheno_onset_pairs.append((pheno, onset))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Problematic JSON string: {prompt_str}")

    output_prompt = json.dumps(prompts_list, indent=4)

    with open("small_text_prompt.json", "w") as out_file:
        out_file.write(output_prompt)
    return prompts_list, pheno_onset_pairs

PROMPT_TEMPLATES_DIR = path.join("prompt_templates")
ONSET_CLASSIFICATION_PROMPT_TEMPLATE = path.join(
    PROMPT_TEMPLATES_DIR, "onset_classification.yml"
)
ONSET_CLASSIFICATION_PROMPTS = path.join(
    PROMPT_TEMPLATES_DIR, "onset_classification_prompts.json"
)

parse_prompt_template(ONSET_CLASSIFICATION_PROMPT_TEMPLATE, ONSET_CLASSIFICATION_PROMPTS)

from openai import OpenAI

MODEL = "gpt-4o"

def test_gpt(prompts):
    client = OpenAI(api_key=OPENAI_KEY, organization=OPENAI_ORG)
    completion = client.chat.completions.create(model="gpt-4o", messages=prompts, seed=9)
    output_text = completion.choices[0].message.content
    print(output_text)
    return output_text

prompts, pheno_onset_pairs = generate_data_prompts(ONSET_CLASSIFICATION_PROMPTS)
print(f"Total prompts generated: {len(prompts)}")
print(f"Total pheno onset pairs: {len(pheno_onset_pairs)}")

output_list = []
#for prompt in prompts[:11]:
for prompt in prompts:
    output = test_gpt(prompt)
    output_list.append(output)

output_str = ""
start_line = True

for pheno_onset_tup, result in zip(pheno_onset_pairs, output_list):
    pheno = pheno_onset_tup[0]
    onset = pheno_onset_tup[1]
    if not start_line:
        output_str += f"\n{pheno}| {onset}| {result}"
    else:
        output_str += f"{pheno}| {onset}| {result}"
        start_line = False

with open("small_dataset_output.txt", "w") as out_file:
    out_file.write(output_str)