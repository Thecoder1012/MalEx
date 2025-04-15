import torch
from torch.utils.data import DataLoader
from dataloader import UGRansomeDataset, get_dataloaders
from model import CNNClassifier
from sklearn.metrics import precision_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import google.generativeai as genai
import json
import os
import logging
from datetime import datetime
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key="")
model_llm = genai.GenerativeModel('gemini-1.5-flash')

# Reverse label map for interpretation
LABEL_MAP = {0: 'Bonet', 1: 'DoS', 2: 'Spam', 3: 'Scan', 4: 'Blacklist', 
             5: 'UDP Scan', 6: 'SSH', 7: 'NerisBonet', 8: 'Port Scanning'}

# Key security-relevant features to include in LLM prompts
BASE_FEATURES = ['Time', 'BTC', 'USD', 'Netflow_Bytes', 'Port', 'Protcol', 'Flag']

# Define knowledge base of malware families and their appropriate classifications
# This mapping indicates which malware families are correctly associated with which threat types
FAMILY_CLASSIFICATION_MAP = {
    'Bonet': ['Mirai', 'Bashlite', 'Qbot', 'Ramnit', 'Pushdo', 'Andromeda'],
    'DoS': ['LOIC', 'HOIC', 'Slowloris', 'R-U-Dead-Yet', 'THC-SSL-DOS'],
    'Spam': ['Cutwail', 'Kelihos', 'Necurs', 'Gamut', 'Onliner'],
    'Scan': ['ZMap', 'Masscan', 'Nmap', 'Sparta', 'Zenmap'],
    'Blacklist': ['Zeus', 'Locky', 'DarkComet', 'NjRat', 'Carbanak'],
    'UDP Scan': ['UDPScan', 'UDPProbe', 'ScanUDP', 'UDPmap'],
    'SSH': ['SSHYT', 'SSHPsychos', 'SSHCure', 'Kippo', 'Cowrie'],
    'NerisBonet': ['Neris', 'Rustock', 'Bobax', 'Lethic'],
    'Port Scanning': ['Unicorn', 'Amap', 'SuperScan', 'PortBunny', 'AngryIP']
}

# Define known ransomware families - these are treated separately
RANSOMWARE_FAMILIES = ['WannaCry', 'Ryuk', 'Cerber', 'Locky', 'CryptoLocker', 
                       'Petya', 'NotPetya', 'BadRabbit', 'GandCrab', 'SamSam']

# File paths for results
RESULTS_DIR = 'llm_results'
GEMINI_RESPONSES_FILE = os.path.join(RESULTS_DIR, 'gemini_responses.json')
PROCESSED_SAMPLES_FILE = os.path.join(RESULTS_DIR, 'processed_samples.csv')
METRICS_FILE = os.path.join(RESULTS_DIR, 'metrics.json')

def get_original_features(csv_file, idx):
    """Retrieve the original row from the CSV for a given index."""
    df = pd.read_csv(csv_file)
    return df.iloc[idx]

def load_processed_samples():
    """Load the list of already processed sample IDs from both CSV and marker files."""
    processed = set()
    
    # Load from CSV file
    if os.path.exists(PROCESSED_SAMPLES_FILE):
        try:
            with open(PROCESSED_SAMPLES_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    processed.add(int(row[0]))
            logger.info(f"Found {len(processed)} processed samples in CSV file")
        except Exception as e:
            logger.error(f"Error loading processed samples from CSV: {str(e)}")
    
    # Load from marker files as backup
    marker_dir = os.path.join(RESULTS_DIR, 'processed_markers')
    if os.path.exists(marker_dir):
        marker_files = [f for f in os.listdir(marker_dir) if f.endswith('.done')]
        for marker in marker_files:
            try:
                sample_id = int(marker.split('_')[1].split('.')[0])
                processed.add(sample_id)
            except:
                pass
        logger.info(f"Found {len(marker_files)} marker files")
    
    # Load from emergency markers
    emergency_markers = [f for f in os.listdir(RESULTS_DIR) if f.startswith('emergency_processed_') and f.endswith('.marker')]
    for marker in emergency_markers:
        try:
            sample_id = int(marker.split('_')[2].split('.')[0])
            processed.add(sample_id)
        except:
            pass
    
    # Load from individual response files as another backup
    response_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('response_') and f.endswith('.json')]
    for resp_file in response_files:
        try:
            sample_id = int(resp_file.split('_')[1].split('.')[0])
            processed.add(sample_id)
        except:
            pass
    
    logger.info(f"Found {len(processed)} total processed samples after checking all sources")
    return processed

def save_processed_sample(sample_id):
    """Save a sample ID to the processed samples file."""
    try:
        file_exists = os.path.exists(PROCESSED_SAMPLES_FILE)
        
        with open(PROCESSED_SAMPLES_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['sample_id', 'timestamp'])
            writer.writerow([sample_id, datetime.now().isoformat()])
            
        # Also create a marker file as a redundant record that this sample was processed
        # This provides a backup mechanism if the CSV file gets corrupted
        marker_dir = os.path.join(RESULTS_DIR, 'processed_markers')
        os.makedirs(marker_dir, exist_ok=True)
        with open(os.path.join(marker_dir, f'sample_{sample_id}.done'), 'w') as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error marking sample {sample_id} as processed: {str(e)}")
        # Emergency marker in case of file system issues
        try:
            with open(os.path.join(RESULTS_DIR, f'emergency_processed_{sample_id}.marker'), 'w') as f:
                f.write(datetime.now().isoformat())
        except:
            pass

def save_gemini_response(response_data):
    """Save a Gemini LLM response to the responses file."""
    try:
        if os.path.exists(GEMINI_RESPONSES_FILE):
            # Load existing responses
            with open(GEMINI_RESPONSES_FILE, 'r') as f:
                responses = json.load(f)
        else:
            # Create new responses list
            responses = []
        
        # Add the new response
        responses.append(response_data)
        
        # Save all responses
        with open(GEMINI_RESPONSES_FILE, 'w') as f:
            json.dump(responses, f, indent=2)
        
        # Also save this individual response to a separate file as a backup
        sample_id = response_data['sample_id']
        single_response_file = os.path.join(RESULTS_DIR, f'response_{sample_id}.json')
        with open(single_response_file, 'w') as f:
            json.dump(response_data, f, indent=2)
            
        logger.info(f"Saved response for sample {sample_id} to main file and individual backup")
    except Exception as e:
        logger.error(f"Error saving response: {str(e)}")
        # Save to an emergency backup file if main save fails
        emergency_file = os.path.join(RESULTS_DIR, f'emergency_response_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
        with open(emergency_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        logger.info(f"Saved emergency backup to {emergency_file}")

def is_family_consistent_with_classification(family, classification):
    """
    Check if a malware family is consistent with the predicted classification.
    
    Args:
        family: The malware family from the dataset
        classification: The predicted threat classification
        
    Returns:
        bool: True if consistent, False otherwise
    """
    # If family is in the list for this classification, it's consistent
    if classification in FAMILY_CLASSIFICATION_MAP and family in FAMILY_CLASSIFICATION_MAP[classification]:
        return True
    
    # Special case for ransomware, which is often misclassified
    if family in RANSOMWARE_FAMILIES:
        # Ransomware might be associated with some classifications
        possibly_ransomware_related = ['Bonet', 'Blacklist']
        if classification in possibly_ransomware_related:
            return False  # Explicitly mark as inconsistent to avoid misleading connections
    
    # If we don't have knowledge about this family, we can't verify consistency
    return False

def create_features_for_prompt(original_row, predicted_class):
    """
    Create a list of features for the prompt, handling the malware family appropriately.
    
    Args:
        original_row: DataFrame row with original features
        predicted_class: The predicted threat class
        
    Returns:
        feature_list: List of feature strings to include in prompt
        context_notes: Additional context about the features to include in prompt
    """
    feature_values = []
    context_notes = []
    
    # Add base features first
    for feature in BASE_FEATURES:
        if feature in original_row:
            feature_values.append(f"- {feature}: {original_row[feature]}")
    
    # Handle Family specially
    if 'Family' in original_row:
        family_value = original_row['Family']
        pred_label = LABEL_MAP[predicted_class]
        
        # Check if family is consistent with classification
        if is_family_consistent_with_classification(family_value, pred_label):
            feature_values.append(f"- Family: {family_value}")
        elif family_value in RANSOMWARE_FAMILIES:
            # For ransomware, add a special note but don't include in features
            context_notes.append(
                f"Note: This traffic has a 'Family' value of '{family_value}', which is a known ransomware, "
                f"not typically associated with the '{pred_label}' threat classification. "
                f"This might indicate a complex attack or a potential misclassification."
            )
    
    return feature_values, context_notes

def inference(max_samples=None):
    """
    Run inference with context-aware LLM explanations.
    
    Args:
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        Metrics dictionary with accuracy, precision, and F1 score
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load list of already processed samples
    processed_samples = load_processed_samples()
    
    logger.info(f"Starting inference with LLM explanations")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load test data
    logger.info("Loading test data...")
    _, _, test_loader = get_dataloaders('data/train.csv', 'data/test.csv', batch_size=1)

    # Load the trained model
    logger.info("Loading trained model...")
    model = CNNClassifier(num_classes=9).to(device)
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()

    test_preds, test_labels = [], []
    processed_count = 0
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for batch_idx, (images, labels) in enumerate(test_bar):
            # Skip if this sample was already processed
            if batch_idx in processed_samples:
                logger.info(f"Skipping sample {batch_idx} (already processed)")
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Calculate confidence score for prediction (softmax)
            softmax = torch.nn.functional.softmax(outputs, dim=1)
            confidence = softmax[0][preds.item()].item() * 100  # Convert to percentage
            
            # Get original features for this sample
            original_row = get_original_features('data/test.csv', batch_idx)
            
            # Get the predicted and true labels
            pred_class = preds.item()
            pred_label = LABEL_MAP[pred_class]
            true_label = LABEL_MAP[labels.item()]
            
            # Create context-aware feature list
            feature_values, context_notes = create_features_for_prompt(original_row, pred_class)
            
            # Prepare prompt for Gemini LLM with token limitations and context awareness
            prompt = (
                f"As a security analyst, explain concisely (max 150 words) why a network traffic sample "
                f"was classified as '{pred_label}' with {confidence:.1f}% confidence based on these features:\n"
                f"{chr(10).join(feature_values)}\n\n"
            )
            
            # Add any context notes about potential inconsistencies
            if context_notes:
                prompt += f"{chr(10).join(context_notes)}\n\n"
            
            # Complete the prompt with the request for indicators and mitigation
            prompt += (
                f"Provide 1-2 specific indicators and 2-3 brief mitigation steps that are "
                f"accurately aligned with the '{pred_label}' threat type."
            )

            logger.info(f"Sending context-aware prompt to Gemini LLM for sample {batch_idx}...")
            try:
                # Call Gemini LLM with constraints to limit response size
                response = model_llm.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=250,  # Limit output tokens 
                        temperature=0.2,        # Lower temperature for more concise responses
                        top_p=0.95,
                        top_k=40,
                    )
                )
                explanation = response.text
                
                # Create response data object
                response_data = {
                    'sample_id': batch_idx,
                    'timestamp': datetime.now().isoformat(),
                    'predicted_label': pred_label,
                    'true_label': true_label,
                    'confidence': float(confidence),
                    'features': {f.split(': ')[0][2:]: f.split(': ')[1] for f in feature_values if ': ' in f},
                    'context_notes': context_notes,
                    'prompt': prompt,
                    'explanation': explanation,
                    'correct': pred_label == true_label,
                    'family_included': any(f.startswith('- Family:') for f in feature_values)
                }
                
                # Save response to file immediately after processing each sample
                save_gemini_response(response_data)
                
                # Mark this sample as processed immediately
                save_processed_sample(batch_idx)
                
                logger.info(f"Saved response and marked sample {batch_idx} as processed")
                
                # Store results for metrics
                test_preds.append(preds.item())
                test_labels.append(labels.item())
                
                # Print results for this sample
                logger.info(f"Sample {batch_idx}: Predicted: {pred_label} ({confidence:.1f}%), True: {true_label}")
                if context_notes:
                    logger.info(f"Context notes: {context_notes}")
                
                processed_count += 1
                if max_samples and processed_count >= max_samples:
                    logger.info(f"Reached maximum number of samples ({max_samples}), stopping")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing sample {batch_idx}: {str(e)}")

    # Calculate overall metrics if we processed any samples
    if test_preds:
        test_acc = 100 * sum([p == l for p, l in zip(test_preds, test_labels)]) / len(test_labels)
        test_prec = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'f1': float(test_f1),
            'samples_processed': processed_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save metrics to file
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics: Accuracy={test_acc:.2f}%, Precision={test_prec:.4f}, F1={test_f1:.4f}")
        return metrics
    else:
        logger.info("No new samples were processed")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with context-aware LLM explanations')
    parser.add_argument('--max-samples', type=int, default=None, 
                        help='Maximum number of samples to process (default: all)')
    args = parser.parse_args()
    
    inference(max_samples=args.max_samples)
