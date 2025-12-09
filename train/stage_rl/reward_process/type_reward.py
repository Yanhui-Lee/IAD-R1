import re
from difflib import SequenceMatcher


class AnomalyRewardCalculator:
    
    def __init__(self):

        self.anomaly_vocabulary = {
            "Contamination": [
                "surface contamination", "stain", "dirt", 
                "impurity", "color anomaly"
            ],
            "Presence of foreign objects": [
                "foreign object", "foreign body", "debris", "contaminant object", 
                "extraneous material", "foreign element", "foreign matter", "unwanted object"
            ],
            "Scratch": [
                "surface scratch", "scratch mark", 
                "linear scratch", "score mark", "linear anomaly"
            ],
            "Missing parts": [
                "missing part", "surface notch", "notch", "gap", "chip", 
                "surface discontinuity"
            ],
            "Deformation": [
                "shape distortion", "warping", "bending", "twisting", 
                "shape deviation", "geometric distortion", "irregularity", "bent component"
            ],
            "Hole": [
                "opening", "perforation", "puncture", "cavity", "void", 
                "aperture", "penetration defect", "through-hole"
            ],
            "Damage": [
                "structural damage", "breakage", "fracture", "rupture", 
                "deterioration", "material damage", "surface damage"
            ],
            "Abrasion": [
                "wear", "grinding damage", "surface erosion", 
                "wear mark", "surface wear"
            ]
        }
        
        self.category_mapping = {
            "Surface Anomalies": ["Contamination", "Presence of foreign objects", "Scratch", "Missing parts"],
            "Structural Anomalies": ["Deformation", "Hole", "Damage", "Abrasion"]
        }

        self.group_vocabulary = {
            "Surface Anomalies": [
                "surface anomalies",
                "surface anomaly",
            ],
            "Structural Anomalies": [
                "structural anomalies",
                "structural anomaly",
            ]
        }


        self.reward_config = {
            'exact_match': 1.0,      # exact match
            'semantic_match': 0.85,  # semantic match
            'category_match': 0.6,   # category match
            'fuzzy_match': 0.4,      # fuzzy match
            'group_match': 0.3,      # group match
            'no_match': 0.0          # else
        }
        
        # fuzzy matching threshold
        self.fuzzy_threshold = 0.7
        
        self._build_lookup_index()
    
    def _build_lookup_index(self):
        self.keyword_to_category = {}
        self.category_to_group = {}
        

        for category, keywords in self.anomaly_vocabulary.items():

            self.keyword_to_category[self._normalize_text(category)] = category
            
            for keyword in keywords:
                self.keyword_to_category[self._normalize_text(keyword)] = category
        

        for group, categories in self.category_mapping.items():
            for category in categories:
                self.category_to_group[category] = group
        

        self.group_keyword_to_group = {}
        for group, terms in self.group_vocabulary.items():
            self.group_keyword_to_group[self._normalize_text(group)] = group

            for t in terms:
                self.group_keyword_to_group[self._normalize_text(t)] = group
    
    def _normalize_text(self, text: str) -> str:

        if not text:
            return ""
        
        text = text.lower().strip()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s-]', '', text)
        
        return text
    
    def _find_best_match(self, text: str) -> tuple:

        normalized_text = self._normalize_text(text)
        
        # exact match
        if normalized_text in self.keyword_to_category:
            return self.keyword_to_category[normalized_text], 1.0
        
        # semantic match
        best_category = None
        best_confidence = 0.0
        
        for keyword, category in self.keyword_to_category.items():
            if normalized_text in keyword or keyword in normalized_text:
                shorter = min(len(normalized_text), len(keyword))
                longer = max(len(normalized_text), len(keyword))
                confidence = shorter / longer
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_category = category
        
        if best_category:
            return best_category, best_confidence
        
        # fuzzy match
        for keyword, category in self.keyword_to_category.items():
            similarity = SequenceMatcher(None, normalized_text, keyword).ratio()
            if similarity >= self.fuzzy_threshold and similarity > best_confidence:
                best_confidence = similarity
                best_category = category
        
        return best_category, best_confidence
    
    def _get_group_from_text(self, text: str):

        if not text:
            return None
        norm = self._normalize_text(text)
        return self.group_keyword_to_group.get(norm)

    
    def compute_reward(self, predicted: str, actual: str) -> float:
        """
        calculate type reward
        
        Args:
            predicted: gpt predict type
            actual: gt type
            
        Returns:
            reward (0.0 - 1.0)
        """
        if not predicted or not actual:
            return self.reward_config['no_match']


        # normalize
        pred_norm = self._normalize_text(predicted)
        actual_norm = self._normalize_text(actual)
        
        pred_group_from_text = self._get_group_from_text(predicted)
        actual_group_from_text = self._get_group_from_text(actual)

        pred_category, pred_conf = self._find_best_match(predicted)
        actual_category, actual_conf = self._find_best_match(actual)

        pred_group_from_category = self.category_to_group.get(pred_category)
        actual_group_from_category = self.category_to_group.get(actual_category)

        final_pred_group = pred_group_from_text or pred_group_from_category
        final_actual_group = actual_group_from_text or actual_group_from_category

        if final_pred_group and final_actual_group and final_pred_group != final_actual_group:
            return self.reward_config['no_match']

        if pred_group_from_text and not actual_group_from_text and final_actual_group == pred_group_from_text:
            return self.reward_config['group_match']
            
        if actual_group_from_text and not pred_group_from_text and final_pred_group == actual_group_from_text:
            return self.reward_config['group_match']

        # 1. exact match
        if pred_norm == actual_norm:
            return self.reward_config['exact_match']
        
        # 2. semantic match
        if pred_norm in actual_norm or actual_norm in pred_norm:
            return self.reward_config['semantic_match']
        
        # 3. find category
        pred_category, pred_confidence = self._find_best_match(predicted)
        actual_category, actual_confidence = self._find_best_match(actual)
        
        if not pred_category or not actual_category:
            # fuzzy match
            similarity = SequenceMatcher(None, pred_norm, actual_norm).ratio()
            if similarity >= self.fuzzy_threshold:
                return similarity * self.reward_config['fuzzy_match']
            return self.reward_config['no_match']
        
        # 4. category match
        if pred_category == actual_category:
            base_score = self.reward_config['category_match']
            confidence_factor = min(pred_confidence, actual_confidence)
            return base_score + (self.reward_config['semantic_match'] - base_score) * confidence_factor
        
        # 5. group match
        pred_group = self.category_to_group.get(pred_category)
        actual_group = self.category_to_group.get(actual_category)
        
        if pred_group and actual_group and pred_group == actual_group:
            return self.reward_config['group_match']
        
        # 6. fuzzy match
        similarity = SequenceMatcher(None, pred_norm, actual_norm).ratio()
        if similarity >= self.fuzzy_threshold:
            return similarity * self.reward_config['fuzzy_match']
        
        return self.reward_config['no_match']
    
    def compute_reward_with_explanation(self, predicted: str, actual: str) -> tuple:
        """
        calculate type reward
        
        Args:
            predicted: gpt predict type
            actual: gt type
            
        Returns:
            reward (0.0 - 1.0)
        """
        if not predicted or not actual:
            return self.reward_config['no_match'], "empty"
        
        pred_norm = self._normalize_text(predicted)
        actual_norm = self._normalize_text(actual)

        pred_group_from_text = self._get_group_from_text(predicted)
        actual_group_from_text = self._get_group_from_text(actual)

        pred_category, pred_conf = self._find_best_match(predicted)
        actual_category, actual_conf = self._find_best_match(actual)

        pred_group_from_category = self.category_to_group.get(pred_category)
        actual_group_from_category = self.category_to_group.get(actual_category)

        final_pred_group = pred_group_from_text or pred_group_from_category
        final_actual_group = actual_group_from_text or actual_group_from_category

        if final_pred_group and final_actual_group and final_pred_group != final_actual_group:
            return self.reward_config['no_match'], f"Predict group: {final_pred_group}，GT group: {final_actual_group}"

        if pred_group_from_text and not actual_group_from_text and final_actual_group == pred_group_from_text:
            return self.reward_config['group_match'], f"Predict group: {pred_group_from_text}，GT category: {actual_category}）"
            

        if actual_group_from_text and not pred_group_from_text and final_pred_group == actual_group_from_text:
            return self.reward_config['group_match'], f"GT group {actual_group_from_text}，Predict category: {pred_category}）"


        # exact match
        if pred_norm == actual_norm:
            return self.reward_config['exact_match'], f"Exact match: '{predicted}' = '{actual}'"
        
        # semantic match
        if pred_norm in actual_norm:
            return self.reward_config['semantic_match'], f"Semantic entailment: '{predicted}' ⊆ '{actual}'"
        if actual_norm in pred_norm:
            return self.reward_config['semantic_match'], f"Semantic entailment: '{actual}' ⊆ '{predicted}'"
        
        # category match
        pred_category, pred_conf = self._find_best_match(predicted)
        actual_category, actual_conf = self._find_best_match(actual)
        
        if not pred_category:
            return self.reward_config['no_match'], f"Unrecognizable predict type: '{predicted}'"
        if not actual_category:
            return self.reward_config['no_match'], f"Unrecognizable gt type: '{actual}'"
        
        if pred_category == actual_category:
            score = self.reward_config['category_match']
            confidence_factor = min(pred_conf, actual_conf)
            final_score = score + (self.reward_config['semantic_match'] - score) * confidence_factor
            return final_score, f"Semantic match: {pred_category} (Confidence: {confidence_factor:.2f})"
        
        # group match
        pred_group = self.category_to_group.get(pred_category)
        actual_group = self.category_to_group.get(actual_category)
        
        if pred_group and actual_group and pred_group == actual_group:
            return self.reward_config['group_match'], f"Category match: {pred_group} ({pred_category} vs {actual_category})"
        
        # fuzzy match
        similarity = SequenceMatcher(None, pred_norm, actual_norm).ratio()
        if similarity >= self.fuzzy_threshold:
            score = similarity * self.reward_config['fuzzy_match']
            return score, f"Fuzzy match: Similarity {similarity:.2f}"
        
        return self.reward_config['no_match'], f"No match: '{predicted}' & '{actual}'"