
import os
import re
import copy
import math

from datetime import datetime

from collections import deque

from reward_process import location_reward , type_reward , description_reward

def consistency_reward(completions, solution, **kwargs):
    pattern_no = r"^(?!.*<location>)(?!.*<type>).*<think>.*?</think><answer>.*?</answer>.*$"  # normal pattern    
    pattern_yes = r".*<think>.*?</think><location>.*?</location><type>.*?</type><answer>.*?</answer>.*"  # abnormal pattern
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # yes or no
        if ground_truth.lower() == "yes":
            # yes --> pattern_yes
            match_result = re.fullmatch(pattern_yes, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
        elif ground_truth.lower() == "no":
            # no --> pattern_no
            match_result = re.fullmatch(pattern_no, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """answer, location, type"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            #resolve ground_truth
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            gt = ground_truth.lower()
            print("ground_truth" + gt)

            # gt == "no"
            if gt == "no":
                print("gpt answer:" + content)
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match:
                    ans = content_match.group(1).strip().lower()
                    print("exact string1:" + ans)
                    if ans == "no":
                        reward = 1.0
            
            # gt == "yes"
            elif gt == "yes":
                print(" ground_truth == 'yes' ")
                total_reward = 0.0
                max_reward = 2.0 
                
                # ----- type reward -----
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                if gpt_type_match and gt_type_match:
                    gpt_type = gpt_type_match.group(1).strip().lower()
                    gt_type = gt_type_match.group(1).strip().lower()
                    type_calculator = type_reward.AnomalyRewardCalculator()
                    type_score = type_calculator.compute_reward(gpt_type, gt_type)
                    total_reward += type_score
                else:
                    pass # reward += 0

                # location reward
                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                if gpt_location_match and gt_location_match:
                    gpt_location = gpt_location_match.group(1).strip().lower()
                    gt_location = gt_location_match.group(1).strip().lower()
                    location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                    total_reward += location_score
                else:
                    pass

                reward = total_reward / max_reward # 0~1

                # answer reward
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match:
                    ans = answer_match.group(1).strip().lower()
                    print("exact string2:" + ans)
                    if ans == "yes":
                        reward += 1.0
                    
                
                
        except Exception:
            pass  # reward = 0   
        rewards.append(reward)            
    return rewards





def consistency_reward_cot(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        

        if model_answer != ground_truth:

            rewards.append(0.0)
            continue
        

        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                consistency_reward = 1.0  
            elif tag_count == 2:
                consistency_reward = 0.7 
            elif tag_count == 1:
                consistency_reward = 0.4 
            else:
                consistency_reward = 0.0 
        else:
            consistency_reward = 0.0
        
        rewards.append(consistency_reward)
    
    return rewards

def format_consistency_reward_cot(completions, solution, **kwargs):
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        
        if model_answer != ground_truth:
            rewards.append(0.0)
            continue
        
        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            format_consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                format_consistency_reward = 1.0
            elif tag_count == 2:
                format_consistency_reward = 0.7
            elif tag_count == 1:
                format_consistency_reward = 0.4
            else:  # tag_count == 0
                format_consistency_reward = 0.0
        else:

            format_consistency_reward = 0.0
        
        rewards.append(format_consistency_reward)
    
    return rewards


def accuracy_reward_cot_wo_type(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
  
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 


                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                gpt_location = gpt_location_match.group(1).strip().lower()
                gt_location = gt_location_match.group(1).strip().lower()
                location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                total_reward += location_score

                reward = total_reward / max_reward


                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                    
                  
                
        except Exception:
            pass 
        rewards.append(reward)            
    return rewards

def accuracy_reward_cot_wo_location(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0
            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 
                
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                gpt_type = gpt_type_match.group(1).strip().lower()
                gt_type = gt_type_match.group(1).strip().lower()
                type_calculator = type_reward.AnomalyRewardCalculator()
                type_score = type_calculator.compute_reward(gpt_type, gt_type)
                total_reward += type_score

                reward = total_reward / max_reward
                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                
        except Exception:
            pass    
        rewards.append(reward)            
    return rewards

def format_reward_cot_base(completions, solution, **kwargs):
    pattern = r".*<think>.*?</think><answer>.*?</answer>.*" 
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        match_result = re.fullmatch(pattern, content, re.DOTALL)
        rewards.append(1.0 if match_result else 0.0)
    return rewards

def accuracy_reward_cot_base(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                answer_match = re.search(r'<answer>(.*?)</answer>', content)

                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0

        except Exception:
            pass  
        rewards.append(reward)            
    return rewards

def wo_format(completions, solution, **kwargs):
    rewards = 0
    return rewards
