# -*- coding: utf-8 -*-
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import argparse
import os
import logging
from datetime import datetime
import torch.nn.functional as F
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def independent_bce_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Independent binary cross-entropy loss for dual-output
    Each output is treated as independent binary classification
    
    Args:
        logits: shape (B, 2) - [passage_logits, graph_logits]
        targets: shape (B, 2) - [passage_label, graph_label]
        alpha: Weight for positive class
        gamma: Focal loss parameter
    """
    # Apply sigmoid to get independent probabilities
    passage_probs = torch.sigmoid(logits[:, 0])  # Independent passage probability
    graph_probs = torch.sigmoid(logits[:, 1])    # Independent graph probability
    
    # Calculate BCE loss for each output independently
    passage_loss = F.binary_cross_entropy(passage_probs, targets[:, 0], reduction='none')
    graph_loss = F.binary_cross_entropy(graph_probs, targets[:, 1], reduction='none')
    
    # Apply focal loss weighting
    passage_p_t = passage_probs * targets[:, 0] + (1 - passage_probs) * (1 - targets[:, 0])
    graph_p_t = graph_probs * targets[:, 1] + (1 - graph_probs) * (1 - targets[:, 1])
    
    passage_mod = (1 - passage_p_t) ** gamma
    graph_mod = (1 - graph_p_t) ** gamma
    
    # Apply alpha weighting
    passage_alpha = (1 - alpha) * (1 - targets[:, 0]) + alpha * targets[:, 0]
    graph_alpha = (1 - alpha) * (1 - targets[:, 1]) + alpha * targets[:, 1]
    
    # Final loss
    passage_loss = passage_alpha * passage_mod * passage_loss
    graph_loss = graph_alpha * graph_mod * graph_loss
    
    # Average of both losses
    total_loss = (passage_loss.mean() + graph_loss.mean()) / 2
    return total_loss

def search_best_threshold(model, dataloader, device, search_steps=0.02, metric='f1', logger=None):
    """
    Threshold search for dual-output with independent sigmoid outputs
    Uses graph_prob > passage_prob as the decision criterion
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Searching threshold"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            labels = batch["label"].to(device)
            logits = model(**inputs).logits  # shape (B, 2)
            # Use independent sigmoid for each output
            passage_probs = torch.sigmoid(logits[:, 0])
            graph_probs = torch.sigmoid(logits[:, 1])
            probs = torch.stack([passage_probs, graph_probs], dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    best_score, best_thr = 0.0, 0.0

    # Use graph_prob > passage_prob as decision criterion
    graph_probs = all_probs[:, 1]  # graph probabilities
    passage_probs = all_probs[:, 0]  # passage probabilities
    binary_labels = (all_labels[:, 0] == 0.0).astype(int)  # Convert to binary labels: 1 if should use graph, 0 if should use passage

    # Search for optimal threshold (graph_prob - passage_prob > threshold)
    for thr in np.arange(0.0, 0.5, search_steps):
        # Decision: graph_prob - passage_prob > threshold
        preds = ((graph_probs - passage_probs) > thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, preds, average='binary', zero_division=0
        )
        
        if metric == 'f1':
            score = f1
        elif metric == 'f1_recall_balance':
            # Balance between F1 and recall
            score = 0.5 * f1 + 0.5 * recall
        
        if score > best_score:
            best_score, best_thr = score, thr

    logger.info(f"Best threshold={best_thr:.2f}, {metric.upper()}={best_score:.4f}")
    return best_thr


class QueryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Dual labels: [passage_prob, graph_prob]
        self.labels = []
        for item in self.data:
            if item["preference"] == "graph":
                self.labels.append([0.0, 1.0])  # [passage=0, graph=1]
            elif item.get("acceptable") == "graph":
                self.labels.append([1.0, 1.0])  # [passage=1, graph=1]
            else:
                self.labels.append([1.0, 0.0])  # [passage=1, graph=0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data[idx]["query"]
        label = self.labels[idx]
        enc = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }


def train_one_epoch(model, dataloader, optimizer, scheduler, device, eval_dataloader=None, eval_steps=50, 
                   save_dir=None, best_f1=0, epoch=None, global_step=0, logger=None):
    model.train()
    total_loss = 0
    step = 0
    current_best_f1 = best_f1
    current_best_threshold = 0.5
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)  # shape (B, 2)
        outputs = model(**inputs)
        logits = outputs.logits  # shape (B, 2)
        loss = independent_bce_loss(logits, labels, alpha=0.25, gamma=2.0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        step += 1
        global_step += 1
        
        # Evaluate every eval_steps using global_step
        if eval_dataloader is not None and global_step % eval_steps == 0:
            logger.info(f"--- Global Step {global_step} Evaluation ---")
            
            # Search for optimal threshold on dev set
            logger.info(f"Searching optimal threshold for step {global_step}...")
            optimal_threshold = search_best_threshold(model, eval_dataloader, device, metric=args.threshold_metric, logger=logger)
            # Evaluate with optimal threshold
            acc, f1, report, graph_recall = evaluate(model, eval_dataloader, device, threshold=optimal_threshold)
            logger.info(f"Global Step {global_step} - Dev Accuracy: {acc:.4f} | F1: {f1:.4f} | Graph Recall: {graph_recall:.4f} (threshold={optimal_threshold:.4f})")
            logger.info(f"Classification Report:\n{report}")
            
            # Save model if performance improved
            if f1 > current_best_f1 and save_dir is not None:
                current_best_f1 = f1
                current_best_threshold = optimal_threshold
                torch.save(model.state_dict(), os.path.join(save_dir, f"best_router_step_{global_step}.pt"))
                logger.info(f"New best model saved! (F1: {f1:.4f}, Epoch: {epoch + 1}, Global Step: {global_step}, Threshold: {optimal_threshold:.4f})")
            
    return total_loss / len(dataloader), current_best_f1, global_step, current_best_threshold


def evaluate(model, dataloader, device, threshold=0.0):
    """
    Evaluation function for dual-output version with independent sigmoid outputs
    Uses graph_prob - passage_prob > threshold as decision criterion
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            batch_labels = batch["label"].to(device)
            outputs = model(**inputs)
            logits = outputs.logits  # shape (B, 2)
            # Use independent sigmoid for each output
            passage_probs = torch.sigmoid(logits[:, 0])
            graph_probs = torch.sigmoid(logits[:, 1])
            # Decision: graph_prob - passage_prob > threshold
            pred_labels = ((graph_probs - passage_probs) > threshold).long()
            preds.extend(pred_labels.cpu().numpy())
            batch_labels_binary = (batch_labels[:, 0] == 0.0).long()  # 1 if should use graph, 0 if should use passage
            labels.extend(batch_labels_binary.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    # Calculate recall for graph class (label=1)
    graph_recall = recall_score(labels, preds, pos_label=1)
    
    report = classification_report(labels, preds, target_names=["passage", "graph"])
    return acc, f1, report, graph_recall


def setup_logger(save_dir):
    """Setup logger for training process"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_log_{timestamp}.log")
    
    logger = logging.getLogger('router_training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def main(args):
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.base_model_path)
    
    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_save_dir = os.path.join(args.save_dir, f"training_{timestamp}")
    os.makedirs(timestamped_save_dir, exist_ok=True)
    
    # Setup logger
    logger, log_file = setup_logger(timestamped_save_dir)
    logger.info(f"Random seed set to: {args.seed}")
    logger.info(f"Starting dual-output router model training, log file: {log_file}")
    logger.info(f"Checkpoint directory: {timestamped_save_dir}")
    logger.info(f"Training parameters: {vars(args)}")

    # Load data
    logger.info("Loading training data...")
    train_dataset = QueryDataset(args.train_path, tokenizer)
    dev_dataset = QueryDataset(args.dev_path, tokenizer)
    test_dataset = QueryDataset(args.test_path, tokenizer)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Dev dataset size: {len(dev_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Analyze label distribution for both outputs
    labels_array = np.array(train_dataset.labels)  # shape: (N, 2)
    passage_labels = labels_array[:, 0]  # passage success labels
    graph_labels = labels_array[:, 1]    # graph success labels
    
    # Count for passage output: passage_success vs passage_failure
    passage_success_count = np.sum(passage_labels > 0.5)
    passage_failure_count = len(passage_labels) - passage_success_count
    
    # Count for graph output: graph_success vs graph_failure
    graph_success_count = np.sum(graph_labels > 0.5)
    graph_failure_count = len(graph_labels) - graph_success_count
    
    # Count for each label combination
    label_counts = {}
    for item in train_dataset.labels:
        key = f"[{item[0]}, {item[1]}]"
        label_counts[key] = label_counts.get(key, 0) + 1
    logger.info(f"Label combination distribution: {label_counts}")
    
    # Calculate weights for balanced sampling considering both outputs
    passage_weights = 1.0 / np.array([passage_failure_count if l < 0.5 else passage_success_count 
                                       for l in passage_labels])
    graph_weights = 1.0 / np.array([graph_failure_count if l < 0.5 else graph_success_count 
                                     for l in graph_labels])
    
    # Use the maximum to ensure both outputs are balanced
    samples_weight = np.maximum(passage_weights, graph_weights)
    
    # Normalize weights to avoid extreme values
    samples_weight = samples_weight / samples_weight.sum() * len(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    
    logger.info(f"Using weighted random sampler with replacement=True")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model definition
    logger.info("Initializing dual-output model...")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        args.base_model_path, 
        num_labels=2
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(0.1 * total_steps), 
                                                num_training_steps=total_steps)
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {int(0.1 * total_steps)}")

    # ======================
    # Training and Validation
    # ======================
    best_f1 = 0
    best_threshold = 0.5
    global_step = 0
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        train_loss, current_best_f1, global_step, current_best_threshold = train_one_epoch(model, train_loader, optimizer, scheduler, device, 
                                                                                          eval_dataloader=dev_loader, eval_steps=args.eval_steps,
                                                                                          save_dir=timestamped_save_dir, best_f1=best_f1, epoch=epoch, 
                                                                                          global_step=global_step, logger=logger)
        logger.info(f"Epoch {epoch + 1} - Train loss: {train_loss:.4f}")
        if current_best_f1 > best_f1:
            best_f1 = current_best_f1
            best_threshold = current_best_threshold

        # Epoch-end evaluation using best threshold from training
        logger.info(f"Epoch {epoch + 1} end evaluation...")
        
        # Search for optimal threshold on dev set
        logger.info(f"Searching optimal threshold for epoch {epoch + 1}...")
        optimal_threshold = search_best_threshold(model, dev_loader, device, metric=args.threshold_metric, logger=logger)
        logger.info(f"Optimal threshold for epoch {epoch + 1}: {optimal_threshold:.4f}")
        
        # Evaluate with optimal threshold
        acc, f1, report, graph_recall = evaluate(model, dev_loader, device, threshold=optimal_threshold)
        
        logger.info(f"Epoch {epoch + 1} - Dev Accuracy: {acc:.4f} | F1: {f1:.4f} | Graph Recall: {graph_recall:.4f} (threshold={optimal_threshold:.4f})")
        logger.info(f"Epoch {epoch + 1} Classification Report:\n{report}")

        # Update best model if performance improved (using global step)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = optimal_threshold
            torch.save(model.state_dict(), os.path.join(timestamped_save_dir, f"best_router_step_{global_step}.pt"))
            logger.info(f"New best model saved! (F1: {f1:.4f}, Epoch: {epoch + 1}, Global Step: {global_step}, Threshold: {optimal_threshold:.4f})")

        # Evaluate on test set for the last epoch
        if epoch == args.epochs - 1:
            logger.info(f"Evaluating last epoch model on test set...")
            test_acc, test_f1, test_report, test_graph_recall = evaluate(model, test_loader, device, threshold=optimal_threshold)
            logger.info(f"Last Epoch Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | Graph Recall: {test_graph_recall:.4f} (threshold={optimal_threshold:.4f})")
            logger.info(f"Last Epoch Test Classification Report:\n{test_report}")

    # Save best threshold information
    threshold_info = {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "timestamp": timestamp
    }
    threshold_file = os.path.join(timestamped_save_dir, "best_threshold.json")
    with open(threshold_file, 'w') as f:
        json.dump(threshold_info, f, indent=2)
    logger.info(f"Best threshold saved to: {threshold_file}")

    # ======================
    # Final Test Evaluation
    # ======================
    logger.info("=== Final Test Evaluation ===")
    
    # Load best threshold
    threshold_file = os.path.join(timestamped_save_dir, "best_threshold.json")
    if os.path.exists(threshold_file):
        with open(threshold_file, 'r') as f:
            threshold_info = json.load(f)
        final_threshold = threshold_info["best_threshold"]
        logger.info(f"Using best threshold: {final_threshold:.4f}")
    
    # Find the best model with highest step number
    import glob
    best_model_files = glob.glob(os.path.join(timestamped_save_dir, "best_router_step_*.pt"))
    if best_model_files:
        # Extract step numbers and find the maximum
        step_numbers = []
        for file_path in best_model_files:
            filename = os.path.basename(file_path)
            step_num = int(filename.split('_')[-1].split('.')[0])
            step_numbers.append((step_num, file_path))
        
        # Sort by step number and get the latest
        step_numbers.sort(key=lambda x: x[0])
        latest_best_model = step_numbers[-1][1]
        latest_step = step_numbers[-1][0]
        logger.info(f"Loading best model from step {latest_step}: {latest_best_model}")
        model.load_state_dict(torch.load(latest_best_model))
    else:
        logger.info("No best model found, using current model state")
    
    logger.info("Starting final test evaluation...")
    acc, f1, report, graph_recall = evaluate(model, test_loader, device, threshold=final_threshold)
    
    logger.info(f"Final Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Graph Recall: {graph_recall:.4f} (threshold={final_threshold:.4f})")
    logger.info(f"Final Test Classification Report:\n{report}")
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="models/huggingface.co/microsoft/deberta-v3-base")
    parser.add_argument("--train_path", type=str, 
                       default="router_training_dataset/router_preference_data_train.json")
    parser.add_argument("--dev_path", type=str, 
                       default="router_training_dataset/router_preference_data_val.json")
    parser.add_argument("--test_path", type=str, 
                       default="router_training_dataset/router_preference_data_test.json")
    parser.add_argument("--save_dir", type=str, default="router")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval_steps", type=int, default=50, 
                       help="Evaluate model every N steps during training")
    parser.add_argument("--threshold_metric", type=str, default="f1_recall_balance", 
                       choices=["f1", "f1_recall_balance"],
                       help="Metric for threshold optimization: f1 or f1_recall_balance")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
