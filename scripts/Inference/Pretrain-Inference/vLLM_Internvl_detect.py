import argparse
import base64
import json
from collections import defaultdict
import math
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import logging
from typing import List, Dict, Optional

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

from transformers import AutoTokenizer

sys.path.append("../../../")
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction

os.environ["HF_HOME"] = "~/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InternVLVLLMQuery(GPT4Query):
    def __init__(self, image_path, text_gt, llm, tokenizer, few_shot=[], visualization=False, 
                 domain_knowledge=None, agent=None, mask_path=None, CoT=None, defect_shot=[], args=None):
        super(InternVLVLLMQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.llm = llm
        self.tokenizer = tokenizer
        self.domain_knowledge = domain_knowledge
        self.agent = agent
        self.mask_path = mask_path
        self.CoT = CoT
        self.defect_shot = defect_shot
        self.args = args
        self.sampling_params = SamplingParams(
            temperature=0.0,  
            max_tokens=128,
            stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None
        )

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            return questions, answers, None

        gpt_answers = []
        
        questions = questions[0:1]
        answers = answers[0:1]
        
        prompt = self.build_prompt(questions)
        
        images = self.prepare_images()

        response = self.generate_response_vllm(prompt, images)
        print(response)
        
        gpt_answer = get_ans(response, questions[0]['options'])
        
        if len(gpt_answer) == 0:
            gpt_answer = response
            logging.error(f"No matching answer at {self.image_path}: {questions}")
        gpt_answers.append(gpt_answer)
        print("gpt_answers: " + gpt_answers[0])

        return questions, answers, gpt_answers

    def prepare_images(self) -> List[Image.Image]:
        images = []
        
        if self.visualization:
            self.visualize_image(cv2.imread(self.image_path))
            for ref_image_path in self.few_shot:
                self.visualize_image(cv2.imread(ref_image_path))
        
        for ref_image_path in self.few_shot:
            images.append(Image.open(ref_image_path).convert('RGB'))
        
        for ref_image_path in self.defect_shot:
            images.append(Image.open(ref_image_path).convert('RGB'))
        
        images.append(Image.open(self.image_path).convert('RGB'))
        
        return images

    def build_prompt(self, conversation) -> str:

        prompt_parts = []
        
        prompt_parts.append(instruction)
        
        if self.few_shot:
            prompt_parts.append(f"Following is/are {len(self.few_shot)} image of normal sample, which can be used as a template to compare the image being queried.")
            for i in range(len(self.few_shot)):
                prompt_parts.append("<image>")
        

        if self.defect_shot:
            prompt_parts.append(f"Following is/are {len(self.defect_shot)} image of defect sample for reference.")
            for i in range(len(self.defect_shot)):
                prompt_parts.append("<image>")
    
        prompt_parts.append("Following is the query image:")

        prompt_parts.append("<image>")
        
        prompt_parts.append("Are there any defects in the test image?")
        prompt_parts.append("Please answer by yes or no.")
        
        prompt = "\n".join(prompt_parts)
        
        return prompt

    def generate_response_vllm(self, prompt: str, images: List[Image.Image]) -> str:

        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images}
            },
            sampling_params=self.sampling_params
        )
        

        return outputs[0].outputs[0].text


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
            response_text = response_text.lower().strip('.').strip()
            if response_text in option_clean or option_clean in response_text:
                return key

        return 'E'
        
    except (AttributeError, TypeError):

        return 'E'


def batch_process_with_vllm(llm, tokenizer, batch_data, args):

    all_prompts = []
    all_images = []
    metadata = []
    
    for item in batch_data:
        image_path = item['image_path']
        text_gt = item['text_gt']
        few_shot = item['few_shot']
        defect_shot = item['defect_shot']
        
        internvl_query = InternVLVLLMQuery(
            image_path=image_path,
            text_gt=text_gt,
            llm=llm,
            tokenizer=tokenizer,
            few_shot=few_shot,
            visualization=False,
            defect_shot=defect_shot,
            args=args
        )
        
        questions, answers = internvl_query.parse_conversation(text_gt)
        if questions and answers:
            questions = questions[0:1]
            answers = answers[0:1]
            
            prompt = internvl_query.build_prompt(questions)
            images = internvl_query.prepare_images()
            
            all_prompts.append(prompt)
            all_images.append(images)
            metadata.append({
                'image_path': image_path,
                'questions': questions,
                'answers': answers,
                'text_gt': text_gt,
                'original_image_path': item['original_image_path']
            })
    
    if not all_prompts:
        return []
    

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None
    )
    

    batch_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images}
        }
        for prompt, images in zip(all_prompts, all_images)
    ]
    

    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    

    results = []
    for output, meta in zip(outputs, metadata):
        response = output.outputs[0].text
        gpt_answer = get_ans(response, meta['questions'][0]['options'])
        
        if not gpt_answer:
            gpt_answer = response
            logging.error(f"No matching answer at {meta['image_path']}: {meta['questions']}")
        
        results.append({
            'image_path': meta['image_path'],
            'questions': meta['questions'],
            'answers': meta['answers'],
            'gpt_answers': [gpt_answer],
            'text_gt': meta['text_gt'],
            'original_image_path': meta['original_image_path']
        })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model_path")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")

    parser.add_argument("--domain_knowledge", action="store_true")
    parser.add_argument("--agent", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--defect_shot", type=int, default=0)
    parser.add_argument("--test_dataset", type=str, default="test_data")
    parser.add_argument("--name", type=str, default="InternVL")
    parser.add_argument("--step", type=int, default=500)

    args = parser.parse_args()

    torch.manual_seed(1234)
    model_path = args.model_path
    print(model_path)
    model_name = os.path.split(model_path.rstrip('/'))[-1]


    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, use_fast=False)
    

    if args.dtype == "bf16":
        dtype = "bfloat16"
    elif args.dtype == "fp16":
        dtype = "float16"
    elif args.dtype == "fp32":
        dtype = "float32"
    else:
        dtype = args.dtype
    

    print("Initializing vLLM with multi-modal support...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 20}, 
    )
    print("vLLM initialized successfully!")

    if args.similar_template:
        model_name += "_Similar_template"
    
    if args.defect_shot >= 1:
        model_name += f"_{args.defect_shot}_defect_shot"

    answers_json_path = f"result/{args.name}/{args.test_dataset}/answers_{args.few_shot_model}_shot_{model_name}_vllm.json"
    if not os.path.exists(f"result/{args.name}/{args.test_dataset}/"):
        os.makedirs(f"result/{args.name}/{args.test_dataset}/")
    print(f"Answers will be saved at {answers_json_path}")

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

    if args.debug:
        random.seed(1)
        sample_keys = random.sample(list(chat_ad.keys()), 1600)
    else:
        sample_keys = chat_ad.keys()

    defect_images = defaultdict(list)
    for image_path in sample_keys:
        dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
        object_name = image_path.split("/")[1]
        defect_name = image_path.split("/")[2]

        defect_key = (dataset_name, object_name, defect_name)
        defect_images[defect_key].append(image_path)

    batch_data = []
    for image_path in tqdm(sample_keys, desc="Preparing data"):
        if image_path in existing_images and not args.reproduce:
            continue
            
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]

        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        
        dataset_name = image_path.split("/")[0].replace("DS-MVTec", "MVTec")
        object_name = image_path.split("/")[1]
        defect_name = image_path.split("/")[2]
        defect_key = (dataset_name, object_name, defect_name)
        
        images_in_defect = defect_images[defect_key]
        defect_shot = random.sample([img for img in images_in_defect if img != image_path],
                                   min(args.defect_shot, len(images_in_defect) - 1))
        rel_defect_shot = [os.path.join(args.data_path, path) for path in defect_shot]
        
        batch_data.append({
            'image_path': rel_image_path,
            'text_gt': text_gt,
            'few_shot': rel_few_shot,
            'defect_shot': rel_defect_shot,
            'original_image_path': image_path
        })

    total_batches = (len(batch_data) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Processing batches", total=total_batches):
        batch = batch_data[i:i + args.batch_size]
        results = batch_process_with_vllm(llm, tokenizer, batch, args)
        
        for result in results:
            questions = result['questions']
            answers = result['answers']
            gpt_answers = result['gpt_answers']
            text_gt = result['text_gt']
            original_path = result['original_image_path']
            
            if gpt_answers is None or len(gpt_answers) != len(answers):
                print(f"Error at {original_path}")
                continue
            
            correct = 0
            for j, answer in enumerate(answers):
                if gpt_answers[j] == answer:
                    correct += 1
            accuracy = correct / len(answers) if answers else 0
            print(f"Accuracy: {accuracy:.2f}")
            
            questions_type = [conversion["type"] for conversion in text_gt["conversation"]]
            
            for q, a, ga, qt in zip(questions, answers, gpt_answers, questions_type):
                answer_entry = {
                    "image": original_path,
                    "question": q,
                    "question_type": qt,
                    "correct_answer": a,
                    "gpt_answer": ga
                }
                all_answers_json.append(answer_entry)
        
        if i % (args.batch_size * 10) == 0 or i + args.batch_size >= len(batch_data):
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)

    caculate_accuracy_mmad(answers_json_path)