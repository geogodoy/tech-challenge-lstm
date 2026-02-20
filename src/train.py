# ═══════════════════════════════════════════════════════════════
# ETAPA 5: Treinamento
# Objetivo: Treinar o modelo ajustando os pesos
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import time

# Importar módulos do projeto
from model import StockLSTM, create_model
from preprocessing import preprocess_data

# ══════════════════════════════════════════════════════════════════
# CONFIGURACOES DE TREINAMENTO
# ══════════════════════════════════════════════════════════════════

# Hiperparâmetros
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = None  # None = usar todos os dados (batch gradient descent)

# Diretórios
MODELS_DIR = Path(__file__).parent.parent / "models"


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: str = None,
    verbose: bool = True
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Treina o modelo LSTM.
    
    Args:
        model: Modelo a ser treinado
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de validação
        epochs: Número de épocas
        learning_rate: Taxa de aprendizado
        device: Dispositivo ('cuda' ou 'cpu')
        verbose: Se True, imprime progresso
        
    Returns:
        Tuple com (modelo treinado, lista de train_losses, lista de val_losses)
    """
    # Configurar dispositivo (GPU se disponivel)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # Mover dados para o dispositivo
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Definir funcao de perda e otimizador
    # MSELoss: Mean Squared Error - ideal para regressao
    criterion = nn.MSELoss()
    
    # Adam: Otimizador adaptativo que ajusta a taxa de aprendizado
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if verbose:
        print(f"\nConfiguracao do treinamento:")
        print(f"   Dispositivo: {device}")
        print(f"   Epocas: {epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Loss Function: MSELoss")
        print(f"   Otimizador: Adam")
        print(f"\n{'='*60}")
        print("Iniciando treinamento...")
        print(f"{'='*60}\n")
    
    # Listas para armazenar historico de perdas
    train_losses = []
    val_losses = []
    
    # Melhor loss para early stopping (opcional)
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Tempo inicial
    start_time = time.time()
    
    # Loop de treinamento
    for epoch in range(epochs):
        # ══════════════════════════════════════════════════════════
        # FASE DE TREINO
        # ══════════════════════════════════════════════════════════
        model.train()  # Ativa dropout e batch normalization
        
        # Forward pass: calcula previsões
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass: calcula gradientes
        optimizer.zero_grad()  # Limpa gradientes anteriores
        loss.backward()        # Calcula gradientes (Backpropagation Through Time)
        optimizer.step()       # Atualiza pesos
        
        # ══════════════════════════════════════════════════════════
        # FASE DE VALIDAÇÃO
        # ══════════════════════════════════════════════════════════
        model.eval()  # Desativa dropout
        with torch.no_grad():  # Não calcula gradientes (economia de memória)
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        # Armazenar perdas
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        # Verificar se é o melhor modelo
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch + 1
        
        # Imprimir progresso a cada 10 épocas
        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                  f'Train Loss: {loss.item():.6f} | '
                  f'Val Loss: {val_loss.item():.6f} | '
                  f'Time: {elapsed:.1f}s')
    
    # Tempo total
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Treinamento concluido!")
        print(f"{'='*60}")
        print(f"\nResumo:")
        print(f"   Tempo total: {total_time:.1f}s ({total_time/epochs:.2f}s/época)")
        print(f"   Train Loss final: {train_losses[-1]:.6f}")
        print(f"   Val Loss final: {val_losses[-1]:.6f}")
        print(f"   Melhor Val Loss: {best_val_loss:.6f} (época {best_epoch})")
    
    return model, train_losses, val_losses


def plot_training_history(
    train_losses: List[float], 
    val_losses: List[float],
    save_path: Path = None
) -> None:
    """
    Plota o histórico de treinamento.
    
    Args:
        train_losses: Lista de perdas de treino
        val_losses: Lista de perdas de validação
        save_path: Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de perdas
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.title('Histórico de Treinamento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de perdas (log scale)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE) - Log Scale')
    plt.title('Histórico de Treinamento (Escala Log)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvo em: {save_path}")
    
    plt.close()


def save_trained_model(
    model: nn.Module, 
    train_losses: List[float],
    val_losses: List[float],
    save_dir: Path = MODELS_DIR
) -> Path:
    """
    Salva o modelo treinado e histórico.
    
    Args:
        model: Modelo treinado
        train_losses: Histórico de perdas de treino
        val_losses: Histórico de perdas de validação
        save_dir: Diretório para salvar
        
    Returns:
        Caminho do arquivo salvo
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo completo (state_dict + config)
    model_path = save_dir / "model_lstm.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }, model_path)
    
    print(f"Modelo salvo em: {model_path}")
    
    return model_path


# ══════════════════════════════════════════════════════════════════
# EXECUCAO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("ETAPA 5: Treinamento do Modelo LSTM")
    print("="*60)
    
    # Carregar dados pre-processados
    print("\nCarregando dados pre-processados...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(save_scaler=False)
    
    # Criar modelo
    print("\nCriando modelo...")
    model = create_model()
    
    # Treinar modelo
    model, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Salvar modelo treinado
    print("\nSalvando modelo...")
    save_trained_model(model, train_losses, val_losses)
    
    # Plotar historico de treinamento
    print("\nGerando graficos...")
    plot_path = MODELS_DIR / "training_history.png"
    plot_training_history(train_losses, val_losses, save_path=plot_path)
    
    # Checkpoint
    print("\n" + "="*60)
    print("CHECKPOINT: Modelo treinado!")
    print("="*60)
