import argparse
import base64
import json
import cv2
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
from tqdm import tqdm
import sys
from openai import OpenAI
from requests import RequestException
from difflib import get_close_matches

sys.path.append("../../../")
from helper.summary import caculate_accuracy_mmad


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"

def get_mime_type(ref_image_path):
    if ref_image_path.lower().endswith(".png"):
        return "image/png"
    elif ref_image_path.lower().endswith((".jpeg", ".jpg")):
        return "image/jpeg"
    return "image/jpeg"


def get_ans(response_text, options=None):
    try:
        gpt_answer = response_text
        if options is None:
            return gpt_answer

        for key, value in options.items():
            if gpt_answer.lower().strip('.') == value.lower().strip('.') or \
               gpt_answer.lower().strip('!') == value.lower().strip('.'):
                return key

        for key, value in options.items():
            option_clean = value.lower().strip('.').strip()
            if response_text in option_clean or option_clean in response_text:
                return key

        return 'E'
    except (AttributeError, TypeError):
        return 'E'


def parse_json(self, response_json):
    if response_json is None:
        print("Response is None, returning empty string")
        return ''
    choices = response_json.get('choices', [])
    if choices:
        message = choices[0].get('message', {})
        caption = message.get('content', '')
        if self.visualization:
            print(f"Caption: {caption}")
        return caption
    return ''

class GPT4Query():
    def __init__(self, image_path, text_gt, few_shot=[], visualization=False):
        self.image_path = image_path
        self.text_gt = text_gt
        self.few_shot = few_shot
        self.max_image_size = (512, 512)
        self.api_time_cost = 0
        self.visualization = visualization
        self.max_retries = 5
        self.parse_json = parse_json

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=DASHSCOPE_BASE_URL,
        )

    def encode_image_to_base64(self, image):
        height, width = image.shape[:2]
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')

    def visualize_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def send_request_to_api(self, payload):
        retries = 0
        retry_delay = 2
        while retries < self.max_retries:
            try:
                before = time.time()
                completion = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=payload["messages"],
                    max_tokens=payload.get("max_tokens", 600),
                )
                self.api_time_cost += time.time() - before

                response_json = {
                    "choices": [
                        {"message": {"content": completion.choices[0].message.content}}
                    ]
                }
                return response_json

            except Exception as e:
                print(f"Request failed ({type(e).__name__}): {e}")
                retries += 1
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

        print(f"Failed after {self.max_retries} retries.")
        return None

    def parse_conversation(self, text_gt):
        Question, Answer = [], []
        for key in text_gt.keys():
            if key.startswith("conversation"):
                conversation = text_gt[key]
                for i, QA in enumerate(conversation):
                    options_items = list(QA['Options'].items())
                    options_text = ""
                    new_answer_key = None
                    for new_key, (original_key, value) in enumerate(options_items):
                        options_text += f"{chr(65 + new_key)}. {value}\n"
                        if QA['Answer'] == original_key:
                            new_answer_key = chr(65 + new_key)
                    option_dict = {chr(65 + new_key): value for new_key, (original_key, value) in enumerate(options_items)}
                    questions_text = QA['Question']
                    Question.append({
                        "type": "text",
                        "text": f"Question {i + 1}: {questions_text} \n{options_text}",
                        "options": option_dict,
                    })
                    if new_answer_key is not None:
                        Answer.append(new_answer_key)
                    else:
                        raise ValueError("Answer key not found after shuffling options.")
                break
        return Question, Answer

    def get_query(self, conversation):
        incontext = ""
        if self.few_shot:
            incontext = f"The first {len(self.few_shot)} image is the normal sample, which can be used as a template to compare."

        incontext_image = []
        for ref_image_path in self.few_shot:
            ref_image = cv2.imread(ref_image_path)
            if self.visualization:
                self.visualize_image(ref_image)
            ref_base64_image = self.encode_image_to_base64(ref_image)
            incontext_image.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ref_base64_image}"}
            })

        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": incontext_image + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{get_mime_type(self.image_path)};base64,{base64_image}"
                            }
                        },
                        {"type": "text", "text": "Are there any defects in the test image?"},
                        {"type": "text", "text": "Directly answer by yes or no."},
                    ]
                }
            ],
            "max_tokens": 600,
        }
        return payload

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if not questions or not answers:
            return questions, answers, None

        questions, answers = questions[0:1], answers[0:1]
        messages = self.get_query(questions)
        response = self.send_request_to_api(messages)

        print("==========GPT ANSWER==========")
        if response is None:
            print("API request failed.")
            return questions, answers, None

        gpt_answer = self.parse_json(self, response)
        print(gpt_answer)
        print("=================================")

        gpt_answer = get_ans(gpt_answer, questions[0]['options'])
        gpt_answers = [gpt_answer]
        return questions, answers, gpt_answers


if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("DASHSCOPE_API_KEY= YOUR APIKey")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=0)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--test_dataset", type=str, default="")
    args = parser.parse_args()

    answers_json_path = f"result/qwen_vl_max/{args.test_dataset}/answers_{args.few_shot_model}_shot.json"
    os.makedirs(os.path.dirname(answers_json_path), exist_ok=True)

    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as f:
            all_answers_json = json.load(f)
    else:
        all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "/mnt/nfs/lyh/Project/IAD-R1/Industrial_test",
        "json_path": f"/mnt/nfs/lyh/Project/IAD-R1/data/Test/{args.test_dataset}_format.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as f:
        chat_ad = json.load(f)

    for data_id, image_path in enumerate(tqdm(chat_ad.keys())):
        if image_path in existing_images and not args.reproduce:
            continue

        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, p) for p in few_shot]

        model = GPT4Query(image_path=rel_image_path, text_gt=text_gt, few_shot=rel_few_shot)
        print(f"Processing image {data_id + 1}/{len(chat_ad)}: {image_path}")

        questions, answers, gpt_answers = model.generate_answer()
        if gpt_answers is None:
            continue

        if len(gpt_answers) != len(answers):
            print(f"Answer length mismatch at {image_path}")
            continue

        correct = sum([1 for i, a in enumerate(answers) if gpt_answers[i] == a])
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}, API time: {model.api_time_cost:.2f}s")

        questions_type = [conv["type"] for conv in text_gt["conversation"]]
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            all_answers_json.append({
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            })

        with open(answers_json_path, "w") as f:
            json.dump(all_answers_json, f, indent=4)

    print(f"Processing completed. Results saved to {answers_json_path}")
    caculate_accuracy_mmad(answers_json_path)
