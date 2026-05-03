import json
import pandas as pd
import matplotlib.pyplot as plt

def parse_experiment_records(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    summaries = []
    for record in data['records']:
        res = {
            "input_size": record['config']['data_params']['input_size'],
            "pixel_range": record['config']['data_params']['pixel_range'],
            "model": record['config']['model_params']['architecture'],
            "lr": record['config']['training_params']['learning_rate'],
            "epochs": record['logs']['epochs'],
            
            "final_test_loss": record['logs']['final_test_metrics']['loss'],
            "final_test_acc": record['logs']['final_test_metrics']['accuracy'],
            "final_test_eer": record['logs']['final_test_metrics']['eer'],
            
            "data_augmentation": record['config']['augmentation_params'],
            
            "filtering_ratio ": (record['config']['filtering_params']['keep_ratio'] ),
            "filtering_method ": (record['config']['filtering_params']['filter_function'] ),
            "filtering_parameters ": (record['config']['filtering_params'] , "see the file for more details"),
        }
        summaries.append(res)
    
    return data['metadata']['experiment_id'], data['metadata']['notes'], pd.DataFrame(summaries)

def plot_training_trends(json_path, record_index=0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['records'][record_index]['logs']['training_history']
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Acc')
    plt.plot(epochs, history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Trend')
    plt.legend()

    # Plot EER (The most important metric for Spoofing Detection)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_eer'], label='Val EER', color='red')
    plt.title('Validation EER Trend')
    plt.xlabel('Epochs')
    plt.ylabel('EER')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_training_trends_2(json_path, record_index=0):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['records'][record_index]['logs']['training_history']
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Train Acc')
    # plt.plot(epochs, history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Trend')
    plt.legend()

    # Plot EER (The most important metric for Spoofing Detection)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='loss', color='red')
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def audit_experiment(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for i, record in enumerate(data['records']):
        train_eer = record['logs']['training_history']['eer']
        
        print(f"--- Audit for Record {i} ---")
        # Alert for the "Zero EER" issue in training logs
        if all(v == 0.0 for v in train_eer):
            print("⚠️ WARNING: Training EER is constant 0.0. Check your EERCallback logic!")
        
        final_eer = record['logs']['final_test_metrics']['eer']
        if final_eer < 0.10:
            print(f"✅ EXCELLENT: Final EER ({final_eer:.2%}) is below the 10% threshold.")
        else:
            print(f"ℹ️ NOTICE: Final EER is {final_eer:.2%}. Consider adjusting augmentation.")