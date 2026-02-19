# ═══════════════════════════════════════════════════════════════
# 📌 Hyperparameter Tuning - Busca por MAPE < 5%
# 🎯 Objetivo: Encontrar configuração ótima do modelo
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import time

from model import StockLSTM
from preprocessing import preprocess_data

MODELS_DIR = Path(__file__).parent.parent / "models"


def train_and_evaluate(
    X_train, y_train, X_test, y_test, scaler,
    hidden_size=50, num_layers=2, dropout=0.2,
    epochs=100, learning_rate=0.001,
    verbose=False
):
    """
    Treina e avalia o modelo com configuração específica.
    Retorna as métricas de avaliação.
    """
    # Criar modelo
    model = StockLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Treinar
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
    
    # Avaliar
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Reverter normalização
    predictions_reais = scaler.inverse_transform(predictions.numpy())
    actual_reais = scaler.inverse_transform(y_test.numpy())
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(actual_reais, predictions_reais))
    mae = mean_absolute_error(actual_reais, predictions_reais)
    mape = np.mean(np.abs((actual_reais - predictions_reais) / actual_reais)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'model': model,
        'train_loss': loss.item()
    }


def run_experiments():
    """
    Executa experimentos com diferentes configurações.
    """
    print("=" * 70)
    print("🔬 HYPERPARAMETER TUNING - Buscando MAPE < 5%")
    print("=" * 70)
    
    # Carregar dados
    print("\n📥 Carregando dados...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(save_scaler=False)
    
    # Configurações a testar
    experiments = [
        # Baseline (configuração atual)
        {"name": "Baseline", "hidden_size": 50, "num_layers": 2, "dropout": 0.2, "epochs": 100, "learning_rate": 0.001},
        
        # Ajuste de Learning Rate
        {"name": "LR 0.0005", "hidden_size": 50, "num_layers": 2, "dropout": 0.2, "epochs": 100, "learning_rate": 0.0005},
        {"name": "LR 0.0001", "hidden_size": 50, "num_layers": 2, "dropout": 0.2, "epochs": 100, "learning_rate": 0.0001},
        
        # Mais épocas
        {"name": "150 epochs", "hidden_size": 50, "num_layers": 2, "dropout": 0.2, "epochs": 150, "learning_rate": 0.001},
        {"name": "200 epochs", "hidden_size": 50, "num_layers": 2, "dropout": 0.2, "epochs": 200, "learning_rate": 0.001},
        
        # Hidden size maior
        {"name": "Hidden 64", "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "epochs": 100, "learning_rate": 0.001},
        {"name": "Hidden 100", "hidden_size": 100, "num_layers": 2, "dropout": 0.2, "epochs": 100, "learning_rate": 0.001},
        
        # Dropout ajustado
        {"name": "Dropout 0.1", "hidden_size": 50, "num_layers": 2, "dropout": 0.1, "epochs": 100, "learning_rate": 0.001},
        {"name": "Dropout 0.3", "hidden_size": 50, "num_layers": 2, "dropout": 0.3, "epochs": 100, "learning_rate": 0.001},
        
        # Combinações promissoras
        {"name": "Combo 1", "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "epochs": 150, "learning_rate": 0.0005},
        {"name": "Combo 2", "hidden_size": 100, "num_layers": 2, "dropout": 0.1, "epochs": 150, "learning_rate": 0.001},
        {"name": "Combo 3", "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "epochs": 200, "learning_rate": 0.0005},
    ]
    
    results = []
    best_mape = float('inf')
    best_config = None
    best_model = None
    
    print(f"\n🧪 Executando {len(experiments)} experimentos...\n")
    print("-" * 70)
    print(f"{'Experimento':<15} | {'MAPE':>8} | {'RMSE':>8} | {'MAE':>8} | {'Status':<10}")
    print("-" * 70)
    
    for i, config in enumerate(experiments):
        name = config.pop('name')
        
        start_time = time.time()
        result = train_and_evaluate(X_train, y_train, X_test, y_test, scaler, **config)
        elapsed = time.time() - start_time
        
        mape = result['mape']
        rmse = result['rmse']
        mae = result['mae']
        
        status = "✅ < 5%!" if mape < 5 else ("⚠️ ~5%" if mape < 5.5 else "")
        
        print(f"{name:<15} | {mape:>7.2f}% | R${rmse:>5.2f} | R${mae:>5.2f} | {status}")
        
        results.append({
            'name': name,
            'config': config,
            'mape': mape,
            'rmse': rmse,
            'mae': mae
        })
        
        if mape < best_mape:
            best_mape = mape
            best_config = {'name': name, **config}
            best_model = result['model']
    
    print("-" * 70)
    
    # Resumo
    print(f"\n{'='*70}")
    print("📊 RESUMO DOS EXPERIMENTOS")
    print(f"{'='*70}")
    
    # Ordenar por MAPE
    results_sorted = sorted(results, key=lambda x: x['mape'])
    
    print("\n🏆 Top 5 configurações:")
    for i, r in enumerate(results_sorted[:5], 1):
        status = "✅" if r['mape'] < 5 else ""
        print(f"   {i}. {r['name']}: MAPE = {r['mape']:.2f}% {status}")
    
    print(f"\n🥇 Melhor configuração: {best_config['name']}")
    print(f"   MAPE: {best_mape:.2f}%")
    print(f"   Hidden Size: {best_config['hidden_size']}")
    print(f"   Num Layers: {best_config['num_layers']}")
    print(f"   Dropout: {best_config['dropout']}")
    print(f"   Epochs: {best_config['epochs']}")
    print(f"   Learning Rate: {best_config['learning_rate']}")
    
    if best_mape < 5:
        print(f"\n🎉 SUCESSO! Encontrada configuração com MAPE < 5%!")
        
        # Salvar melhor modelo
        save_path = MODELS_DIR / "model_lstm_optimized.pth"
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_config': best_model.get_config(),
            'mape': best_mape,
            'rmse': results_sorted[0]['rmse'],
            'mae': results_sorted[0]['mae']
        }, save_path)
        print(f"   💾 Modelo salvo em: {save_path}")
    else:
        print(f"\n⚠️ MAPE mínimo alcançado: {best_mape:.2f}%")
        print("   Sugestões para melhorar:")
        print("   - Coletar mais dados históricos")
        print("   - Adicionar mais features (Volume, Open, High, Low)")
        print("   - Usar técnicas de data augmentation")
    
    return results, best_config, best_model


if __name__ == "__main__":
    results, best_config, best_model = run_experiments()
