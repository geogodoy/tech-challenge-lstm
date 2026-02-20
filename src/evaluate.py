# ═══════════════════════════════════════════════════════════════
# ETAPA 6: Avaliacao do Modelo
# Objetivo: Calcular metricas e avaliar performance
# Referencia: GUIA_TREINAMENTO_E_AVALIACAO.md - Parte 2
# ═══════════════════════════════════════════════════════════════

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

from model import StockLSTM
from preprocessing import preprocess_data

# Diretórios
MODELS_DIR = Path(__file__).parent.parent / "models"


def load_trained_model(model_path: Path = None) -> tuple:
    """
    Carrega o modelo treinado e suas configurações.
    
    Returns:
        Tuple com (model, checkpoint)
    """
    if model_path is None:
        model_path = MODELS_DIR / "model_lstm.pth"
    
    print(f"Carregando modelo de: {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    model = StockLSTM(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modelo carregado!")
    print(f"   Train Loss final: {checkpoint['final_train_loss']:.6f}")
    print(f"   Val Loss final:   {checkpoint['final_val_loss']:.6f}")
    
    return model, checkpoint


def make_predictions(model, X_test, scaler):
    """
    Faz previsões e reverte a normalização.
    
    Returns:
        Tuple com (predictions_reais, actual_reais) em R$
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    predictions_np = predictions.numpy()
    
    return predictions_np


def calculate_metrics(actual_reais, predictions_reais):
    """
    Calcula métricas de avaliação.
    
    Returns:
        Dict com MSE, RMSE, MAE, MAPE
    """
    mse = mean_squared_error(actual_reais, predictions_reais)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_reais, predictions_reais)
    mape = np.mean(np.abs((actual_reais - predictions_reais) / actual_reais)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def plot_predictions(actual_reais, predictions_reais, save_path=None):
    """
    Plota gráfico de previsões vs valores reais.
    """
    plt.figure(figsize=(14, 6))
    
    # Gráfico 1: Comparação temporal
    plt.subplot(1, 2, 1)
    n_samples = min(100, len(actual_reais))
    plt.plot(actual_reais[-n_samples:], label='Real', color='blue', linewidth=2)
    plt.plot(predictions_reais[-n_samples:], label='Previsto', color='red', 
             linewidth=2, linestyle='--')
    plt.title(f'Previsão vs Valor Real (Últimas {n_samples} amostras)')
    plt.xlabel('Amostra')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(actual_reais, predictions_reais, alpha=0.5, s=20)
    min_val = min(actual_reais.min(), predictions_reais.min())
    max_val = max(actual_reais.max(), predictions_reais.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    plt.xlabel('Valor Real (R$)')
    plt.ylabel('Valor Previsto (R$)')
    plt.title('Correlação: Previsto vs Real')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvo em: {save_path}")
    
    plt.close()


def diagnose_model(mape):
    """
    Diagnostico baseado no MAPE.
    """
    print(f"\nDIAGNOSTICO:")
    if mape < 5:
        print("   Excelente! Modelo muito preciso.")
        return "excelente"
    elif mape < 10:
        print("   Bom! Modelo com boa precisao.")
        return "bom"
    elif mape < 20:
        print("   Aceitavel. Considere ajustes para melhorar.")
        return "aceitavel"
    else:
        print("   Precisa melhorar. Revise hiperparametros e dados.")
        return "ruim"


def evaluate_model():
    """
    Pipeline completa de avaliacao.
    """
    print("=" * 60)
    print("ETAPA 6: Avaliacao do Modelo LSTM")
    print("=" * 60)
    
    # 1. Carregar modelo
    print("\nCarregando modelo treinado...")
    model, checkpoint = load_trained_model()
    
    # 2. Carregar dados
    print("\nCarregando dados de teste...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(save_scaler=False)
    print(f"   Amostras de teste: {len(X_test)}")
    
    # 3. Fazer previsoes
    print("\nFazendo previsoes...")
    predictions_np = make_predictions(model, X_test, scaler)
    
    # 4. Reverter normalização para R$
    predictions_reais = scaler.inverse_transform(predictions_np)
    actual_reais = scaler.inverse_transform(y_test.numpy())
    
    print(f"   Previsoes feitas: {len(predictions_reais)} amostras")
    
    # 5. Exibir exemplos
    print(f"\nExemplos de previsoes:")
    print(f"   {'Previsto':>12} | {'Real':>12} | {'Erro':>12}")
    print("   " + "-" * 42)
    for i in range(5):
        prev = predictions_reais[i][0]
        real = actual_reais[i][0]
        erro = abs(prev - real)
        print(f"   R$ {prev:>9.2f} | R$ {real:>9.2f} | R$ {erro:>9.2f}")
    
    # 6. Calcular metricas
    print("\n" + "=" * 50)
    print("METRICAS DE AVALIACAO")
    print("=" * 50)
    
    metrics = calculate_metrics(actual_reais, predictions_reais)
    
    print(f"MSE  (Mean Squared Error):     {metrics['mse']:.4f}")
    print(f"RMSE (Root Mean Squared Error): R$ {metrics['rmse']:.2f}")
    print(f"MAE  (Mean Absolute Error):     R$ {metrics['mae']:.2f}")
    print(f"MAPE (Mean Absolute % Error):   {metrics['mape']:.2f}%")
    print("=" * 50)
    
    # 7. Diagnóstico
    diagnosis = diagnose_model(metrics['mape'])
    
    # 8. Gerar graficos
    print("\nGerando graficos...")
    plot_path = MODELS_DIR / "predictions_vs_actual.png"
    plot_predictions(actual_reais, predictions_reais, save_path=plot_path)
    
    # 9. Resumo final
    print("\n" + "=" * 60)
    print("AVALIACAO CONCLUIDA!")
    print("=" * 60)
    print(f"\nResumo:")
    print(f"   RMSE: R$ {metrics['rmse']:.2f} (erro médio em reais)")
    print(f"   MAPE: {metrics['mape']:.2f}% (erro percentual médio)")
    print(f"   Status: {diagnosis.upper()}")
    
    return metrics, diagnosis


# ══════════════════════════════════════════════════════════════════
# EXECUCAO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    metrics, diagnosis = evaluate_model()
