import re
import numpy as np

# Arquivo com os resultados
filename = "result_1.txt"

# Dicionário para armazenar listas de valores
metrics = {
    "accuracy_score": [],
    "f1_macro": [],
    "prediction_time": [],
    "training_time": []
}

# Expressões regulares para capturar os valores
patterns = {
    "accuracy_score": re.compile(r"-- accuracy_score --\s*([\d\.]+)"),
    "f1_macro": re.compile(r"-- f1_macro --\s*([\d\.]+)"),
    "prediction_time": re.compile(r"-- prediction_time --\s*([\d\.]+)"),
    "training_time": re.compile(r"-- training_time --\s*([\d\.]+)")
}

# Ler o arquivo e extrair os valores
with open(filename, "r", encoding="utf-8") as f:
    content = f.read()
    for key, pattern in patterns.items():
        matches = pattern.findall(content)
        metrics[key].extend([float(m) for m in matches])

# Calcular média e desvio padrão
for key, values in metrics.items():
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # desvio padrão amostral
    print(f"{key}: média = {mean:.4f}, desvio padrão = {std:.4f}")