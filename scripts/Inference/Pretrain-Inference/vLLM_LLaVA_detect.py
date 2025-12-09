import argparse
import base64
import json
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import re
import pandas as pd
import torch
from tqdm import tqdm
import sys
import seaborn as sns
import logging
from typing import List, Dict, Optional
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

from transformers import AutoTokenizer, AutoProcessor

sys.path.append("../../../")
from helper.summary import caculate_accuracy_mmad
from GPT4.gpt4v import GPT4Query, instruction


class LLaVAVLLMQuery(GPT4Query):
    def __init__(self, image_path, text_gt, llm, processor, few_shot=[], visualization=False, args=None):
        super(LLaVAVLLMQuery, self).__init__(image_path, text_gt, few_shot, visualization)
        self.llm = llm
        self.processor = processor
        self.args = args
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            stop_token_ids=[processor.tokenizer.eos_token_id] if hasattr(processor, 'tokenizer') else None
        )

    def generate_answer(self):
        questions, answers = self.parse_conversation(self.text_gt)
        if questions == [] or answers == []:
            print(f"Warning: No questions/answers found for {self.image_path}")
            return questions, answers, None

        gpt_answers = []
        
        questions = questions[0:1]
        answers = answers[0:1]

        prompt = self.build_prompt(questions)
        
        images = self.prepare_images()
        
        try:
            response = self.generate_response_vllm(prompt, images)
            print(f"Response for {os.path.basename(self.image_path)}: {response[:100]}...")
            
            gpt_answer = get_ans(response, questions[0]['options'])
            
            if not gpt_answer:
                gpt_answer = response
                logging.error(f"No matching answer at {self.image_path}: {questions}")
            
            gpt_answers.append(gpt_answer)
            print(f"Extracted answer: {gpt_answer}")
            
        except Exception as e:
            print(f"Error generating response for {self.image_path}: {e}")
            return questions, answers, None

        return questions, answers, gpt_answers

    def prepare_images(self) -> List[Image.Image]:
        images = []
        
        for ref_image_path in self.few_shot:
            if self.visualization:
                ref_image = cv2.imread(ref_image_path)
                self.visualize_image(ref_image)

            images.append(Image.open(ref_image_path).convert("RGB"))
        
        images.append(Image.open(self.image_path).convert("RGB"))
        
        return images

    def build_prompt(self, conversation) -> str:

        messages = []
        
        content_parts = []
        
        if self.few_shot:
            content_parts.append({
                "type": "text",
                "text": f"Following is {len(self.few_shot)} image of normal sample, "
                        "which can be used as a template to compare the image being queried."
            })
            for i in range(len(self.few_shot)):
                content_parts.append({"type": "image"})
        
        content_parts.append({
            "type": "text",
            "text": "Following is image of test sample:"
        })
        content_parts.append({"type": "image"})
        
        content_parts.append({
            "type": "text",
            "text": "Are there any defects in the query image?"
        })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
        
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
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

        ans_match = re.search(r'<answer>(.*?)</answer>', response_text)
        gpt_answer = ans_match.group(1).strip().lower()  # 转为小写便于比较
        
        if options is None:
            return gpt_answer
        
        for key, value in options.items():
            if gpt_answer == value.lower().strip('.'):
                return key
        
        for key, value in options.items():
            option_clean = value.lower().strip('.').strip()
            if gpt_answer in option_clean or option_clean in gpt_answer:
                return key
        
        return 'E'
        
    except (AttributeError, TypeError):

        return 'E'


def batch_process_with_vllm(llm, processor, batch_data, args):

    all_prompts = []
    all_images = []
    metadata = []
    
    for item in batch_data:
        image_path = item['image_path']
        text_gt = item['text_gt']
        few_shot = item['few_shot']
        
        llava_query = LLaVAVLLMQuery(
            image_path=image_path,
            text_gt=text_gt,
            llm=llm,
            processor=processor,
            few_shot=few_shot,
            visualization=False,
            args=args
        )
        
        questions, answers = llava_query.parse_conversation(text_gt)
        if questions and answers:
            questions = questions[0:1]
            answers = answers[0:1]
            
            prompt = llava_query.build_prompt(questions)
            images = llava_query.prepare_images()
            
            all_prompts.append(prompt)
            all_images.append(images)
            metadata.append({
                'image_path': image_path,
                'questions': questions,
                'answers': answers,
                'text_gt': text_gt
            })
    
    if not all_prompts:
        return []
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop_token_ids=[processor.tokenizer.eos_token_id] if hasattr(processor, 'tokenizer') else None
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
            'text_gt': meta['text_gt']
        })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                        default="model_path")
    parser.add_argument("--few_shot_model", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--similar_template", action="store_true")
    parser.add_argument("--record_history", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--test_dataset", type=str, default="test_data")
    parser.add_argument("--name", type=str, default="LLaVA")
    
    args = parser.parse_args()
    
    torch.manual_seed(6666)
    model_path = args.model_path
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print("Initializing vLLM with multi-modal support...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=32768,  
        enforce_eager=True,
        limit_mm_per_prompt={"image": 10}, 
    )
    print("vLLM initialized successfully!")
    
    model_name = os.path.split(model_path.rstrip('/'))[-1]
    if args.similar_template:
        model_name = model_name + "_Similar_template"
    
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
    
    batch_data = []
    for image_path in tqdm(chat_ad.keys(), desc="Preparing data"):
        if image_path in existing_images and not args.reproduce:
            continue
            
        text_gt = chat_ad[image_path]
        if args.similar_template:
            few_shot = text_gt["similar_templates"][:args.few_shot_model]
        else:
            few_shot = text_gt["random_templates"][:args.few_shot_model]
        
        rel_image_path = os.path.join(args.data_path, image_path)
        rel_few_shot = [os.path.join(args.data_path, path) for path in few_shot]
        
        batch_data.append({
            'image_path': rel_image_path,
            'text_gt': text_gt,
            'few_shot': rel_few_shot,
            'original_image_path': image_path
        })
    
    total_batches = (len(batch_data) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Processing batches", total=total_batches):
        batch = batch_data[i:i + args.batch_size]
        results = batch_process_with_vllm(llm, processor, batch, args)
        
        for result in results:
            questions = result['questions']
            answers = result['answers']
            gpt_answers = result['gpt_answers']
            text_gt = result['text_gt']
            image_path = result['image_path']
            
            original_path = None
            for item in batch:
                if item['image_path'] == image_path:
                    original_path = item['original_image_path']
                    break
            
            if gpt_answers is None or len(gpt_answers) != len(answers):
                print(f"Error at {image_path}")
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
            
            with open(answers_json_path, "w") as file:
                json.dump(all_answers_json, file, indent=4)
    
    caculate_accuracy_mmad(answers_json_path)