def map_location_to_region(gpt_location , gt_location):
    """
    compare location gt and gpt answer
    """
    # Preprocess gpt answer: convert to lowercase, remove redundant words
    text_gpt = gpt_location.lower().strip()
    # The initial position is at the center (5)
    position_gpt = 5

    # level
    if "left" in text_gpt:
        position_gpt -= 1  
    elif "right" in text_gpt:
        position_gpt += 1  
    
    # vertial
    if "top" in text_gpt or "upper" in text_gpt:
        position_gpt -= 3  
    elif "bottom" in text_gpt or "lower" in text_gpt:
        position_gpt += 3  
    # 1~9
    position_gpt = max(1, min(9, position_gpt))


    """
    Preprocess gt
    """
    text_gt = gt_location.lower().strip()
    # The initial position is at the center (5)
    position_gt = 5
    # level
    if "left" in text_gt:
        position_gt -= 1
    elif "right" in text_gt:
        position_gt += 1
    
    # vertical
    if "top" in text_gt or "upper" in text_gt:
        position_gt -= 3
    elif "bottom" in text_gt or "lower" in text_gt:
        position_gt += 3
    # 1~9
    position_gt = max(1, min(9, position_gt))


    if position_gpt == position_gt:
        return 1
    else:
        return 0
    