import argparse
import base64
import json
import requests
import cv2
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
from requests import RequestException
from tqdm import tqdm
import sys
import re
from difflib import get_close_matches
import concurrent.futures

sys.path.append("../../../")

from helper.summary import caculate_accuracy_mmad

error_keywords = ['please', 'sorry', 'today', 'cannot assist']

# Your openai api key
API_KEYS = [
    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]

class GPT4Query():
    def __init__(self, image_path, text_gt, few_shot=[], visualization=False, api_key_index=0):
        self.api_key_index = api_key_index
        self.api_key = API_KEYS[api_key_index]
        self.url = "https://api.openai.com/v1/chat/completions"
        self.image_path = image_path
        self.text_gt = text_gt
        self.few_shot = few_shot
        self.max_image_size = (512, 512)
        self.api_time_cost = 0
        self.visualization = visualization
        self.max_retries = 5
        self.parse_json = parse_json

    def encode_image_to_base64(self, image):
        height, width = image.shape[:2]
        scale = min(self.max_image_size[0] / width, self.max_image_size[1] / height)
        new_width, new_height = int(width * scale), int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, encoded_image = cv2.imencode('.jpg', resized_image)
        return base64.b64encode(encoded_image).decode('utf-8')

    def visualize_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()

    def send_request_to_api(self, payload):
        max_retries = self.max_retries
        retry_delay = 1
        retries = 0

        while retries < max_retries:
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            try:
                before = time.time()
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    response_json = response.json()
                    choices = response_json.get('choices', [])
                    
                    if choices:
                        content = choices[0]['message']['content']
                        if any(word in content.lower() for word in error_keywords):
                            print(f"Error response for {self.image_path}: {content}")
                            retries += 1
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 2, 60)
                            continue
                        
                        self.api_time_cost += time.time() - before
                        return response_json
                    else:
                        print(f"No choices in response with API key #{self.api_key_index}: {response_json}")
                        retries += 1
                
                else:
                    print(f"HTTP Error {response.status_code} with API key #{self.api_key_index}: {response.text}")
                    retries += 1
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)

            except requests.exceptions.Timeout:
                print(f"Request timeout for {self.image_path} with API key #{self.api_key_index}, retrying...")
                retries += 1
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                
            except RequestException as e:
                print(f"Request failed with API key #{self.api_key_index}: {e}, retrying...")
                retries += 1
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                
            except Exception as e:
                print(f"Unexpected error with API key #{self.api_key_index}: {e}")
                retries += 1
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

        print(f"Failed to send request after {max_retries} retries")
        return None

    def parse_conversation(self, text_gt):
        Question = []
        Answer = []
        keyword = "conversation"

        for key in text_gt.keys():
            if key.startswith(keyword):
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

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None
        
        questions = questions[0:1]
        answers = answers[0:1]
        messages = self.get_query(questions)
        response = self.send_request_to_api(messages)
        
        print("==========GPT ANSWER==========")
        
        if response is None:
            print("API request failed after trying all available API keys, skipping this image.")
            return questions, answers, None
        
        gpt_answer = self.parse_json(self, response)
        print(gpt_answer)
        print("==========GPT ANSWER==========")
        
        gpt_answer = get_ans(gpt_answer, questions[0]['options'])
        gpt_answers = []
        if len(gpt_answer) == 0:
            gpt_answer.append(response)
        gpt_answers.append(gpt_answer)

        return questions, answers, gpt_answers

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
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ref_base64_image}",
                    "detail": "low"
                }
            })
        
        image = cv2.imread(self.image_path)
        if self.visualization:
            self.visualize_image(image)
        base64_image = self.encode_image_to_base64(image)

        payload = {
            "model": "gpt-4.1-nano",
            "messages": [
                {
                    "role": "user",
                    "content": incontext_image + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{get_mime_type(self.image_path)};base64,{base64_image}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Are there any defects in the test image?"
                        },
                        {
                            "type": "text",
                            "text": "Directly answer by yes or no."
                        },
                    ]
                }
            ],
            "max_tokens": 600,
        }
        return payload

def get_mime_type(ref_image_path):
    if ref_image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif ref_image_path.lower().endswith(".jpeg") or ref_image_path.lower().endswith(".jpg"):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/jpeg"
    return mime_type

def get_ans(response_text, options=None):
    try:
        gpt_answer = response_text

        if options is None:
            return gpt_answer
        
        for key, value in options.items():
            if gpt_answer.lower().strip('.') == value.lower().strip('.') or gpt_answer.lower().strip('!') == value.lower().strip('.'):
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
        first_choice = choices[0]
        message = first_choice.get('message', {})
        caption = message.get('content', '')
        
        if self.visualization:
            print(f"Caption: {caption}")
        return caption

    return ''

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot_model", type=int, default=0)
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--test_dataset", type=str, default="")
    args = parser.parse_args()
    
    model_name = "gpt-4.1-nano"
    if args.similar_template:
        model_name += "_Similar_template"
    answers_json_path = f"result/gpt_4.1_nano/{args.test_dataset}/answers_{args.few_shot_model}_shot_{model_name}.json"

    if os.path.exists(answers_json_path):
        with open(answers_json_path, "r") as file:
            all_answers_json = json.load(file)
    else:
        all_answers_json = []

    existing_images = [a["image"] for a in all_answers_json]

    cfg = {
        "data_path": "/mnt/nfs/lyh/Project/IAD-R1/Industrial_test",
        "json_path": f"/mnt/nfs/lyh/Project/IAD-R1/data/Test/{args.test_dataset}_format.json"
    }
    args.data_path = cfg["data_path"]

    with open(cfg["json_path"], "r") as file:
        chat_ad = json.load(file)

    api_key_index = 0

    for data_id, image_path in enumerate(tqdm(chat_ad.keys())):
        if image_path in existing_images and not args.reproduce:
            continue
            
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        
        model = GPT4Query(image_path=rel_image_path, text_gt=text_gt, few_shot=rel_few_shot, api_key_index=api_key_index)
        
        
        print(f"Processing image {data_id + 1}/{len(chat_ad)} with API key #{model.api_key_index}")
        
        questions, answers, gpt_answers = model.generate_answer()
        
        if gpt_answers is None:
            print(f"Skipping {image_path} due to API failure after trying all keys")
            continue
            
        if len(gpt_answers) != len(answers):
            print(f"Answer length mismatch at {image_path}")
            continue
            
        correct = 0
        for i, answer in enumerate(answers):
            if gpt_answers[i] == answer:
                correct += 1
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")
        print(f"API time cost: {model.api_time_cost:.2f}s")

        questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
        
        # Update answer records
        for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
            answer_entry = {
                "image": image_path,
                "question": q,
                "question_type": qt,
                "correct_answer": a,
                "gpt_answer": ga
            }
            all_answers_json.append(answer_entry)
        with open(answers_json_path, "w") as file:
            json.dump(all_answers_json, file, indent=4)

    print(f"Processing completed. Final results saved to {answers_json_path}")
    caculate_accuracy_mmad(answers_json_path)