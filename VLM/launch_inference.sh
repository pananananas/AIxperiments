#!/bin/bash

MODEL_PATH="weights"
PROMPT="What kind of tattoo is that? What Does it present and in what style. This tattoo presents a "

# PROMPT="Describe this image. The output must be structured as JSON. I want you to answer questions: Does this photo contain a tattoo? What is the tattoo? Is the tattoo visible? What is the color of the tattoo? What is the object of the tattoo? What is the style of the tattoo? The output must be structured like this: {'contains_tattoo': True, 'tattoo': 'butterfly', 'tattoo_visible': True, 'tattoo_color': 'black', 'tattoo_object': 'butterfly', 'tattoo_style': 'realistic', 'tattoo_description': text}"

IMAGE_FILE_PATH="example_img.jpeg"
MAX_TOKENS_TO_GENERATE=1000
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
