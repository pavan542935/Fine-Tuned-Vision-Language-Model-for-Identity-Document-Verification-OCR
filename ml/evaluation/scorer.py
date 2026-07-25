import json
from pathlib import Path
import difflib

def extract_json_from_text(text):
    # Robustly find first { and last } to handle conversational filler from base model
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except:
            return None
    return None

def fuzzy_match_score(str1, str2):
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str(str1).lower(), str(str2).lower()).ratio()

def score_predictions():
    pred_file = Path("ml/evaluation/baseline_predictions.json")
    if not pred_file.exists():
        raise FileNotFoundError(f"{pred_file} not found. Run inference_base.py first.")
        
    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
        
    fields = ["Name", "DOB", "ID Number", "Address"]
    
    metrics = {f: {"exact": 0, "fuzzy_sum": 0.0} for f in fields}
    parse_failures = 0
    total = len(predictions)
    
    for p in predictions:
        gt = p["ground_truth"]
        raw_out = p["raw_output"]
        
        pred_json = extract_json_from_text(raw_out)
        if pred_json is None:
            parse_failures += 1
            pred_json = {} # count all fields as missed
            
        for f in fields:
            gt_val = gt.get(f, "")
            pred_val = pred_json.get(f, "")
            
            # Exact
            if str(gt_val).strip() == str(pred_val).strip():
                metrics[f]["exact"] += 1
                
            # Fuzzy
            metrics[f]["fuzzy_sum"] += fuzzy_match_score(gt_val, pred_val)
            
    # Compile report
    report = f"# Baseline Evaluation Report\n\n"
    report += f"Total Samples: {total}\n"
    report += f"Parse Failures (Raw output wasn't valid JSON): {parse_failures} / {total} ({(parse_failures/total)*100:.1f}%)\n\n"
    
    report += "| Field | Exact Match Accuracy | Fuzzy Match Score (Avg) |\n"
    report += "|-------|----------------------|-------------------------|\n"
    
    for f in fields:
        exact_acc = (metrics[f]["exact"] / total) * 100
        fuzzy_avg = (metrics[f]["fuzzy_sum"] / total) * 100
        report += f"| {f} | {exact_acc:.1f}% | {fuzzy_avg:.1f}% |\n"
        
    report_path = Path("ml/evaluation/baseline_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(report)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    score_predictions()
