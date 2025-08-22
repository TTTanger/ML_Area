import pandas as pd
import math

def empirical_conditional_entropy(df, key, condition):
    data = df[key]
    conditions = []
    h = 0
    entropy = 0
    for label in df:
        if df[condition] not in conditions:
            conditions.append(df[condition])
    # # Count the length
    # data_length = len(data)
    labels_dict = {}
    label_condition_dict = {}

    # How many times the label occurs
    for label in data:
        if label not in labels_dict:
            labels_dict[label] = 1
        else:
            labels_dict[label] += 1

    # How many times the label occurs when the condition is met
    for label, cond in zip(df[key], df[condition]):
        if label + "_" + cond not in label_condition_dict:
            label_condition_dict[label + "_" + cond] = 1
        else: 
            label_condition_dict[label + "_" + cond] += 1
    
    
    for dict_key, value in label_condition_dict.items():
        raw_key = dict_key.split('_')[0]
        h = - value / labels_dict[raw_key]*math.log2(value / labels_dict[raw_key])
        entropy += labels_dict[raw_key] / len(df) * h
    
    return entropy
    
def shannon_entropy(data: list):
    data_length = len(data)
    label_count_dict = {}
    result = 0
    for label in data:
        if label not in label_count_dict:
            label_count_dict[label] = 1
        else:
            label_count_dict[label] += 1
    for key in label_count_dict:
        result += - label_count_dict[key] / data_length * math.log2(label_count_dict[key] / data_length)
    return result

def gain(df: pd.DataFrame, key, condition, data):
    return shannon_entropy(data) - empirical_conditional_entropy(df, key, condition)

if __name__ == "__main__":
    data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal"],
    "Windy": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak"],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes"]
}

    df = pd.DataFrame(data)
    print(df)

    entropy = shannon_entropy(data["Play"]) 
    print(entropy)

    empirical_conditional_entropy_val = empirical_conditional_entropy(df, "Outlook", "Play")
    print(empirical_conditional_entropy_val)

    gain_val = gain(df, "Outlook", "Play", data["Play"])
    print(gain_val)

