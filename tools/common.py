import string
import torch

class EngineDescpriptor:
    def __init__(self, name, version, autor):
        self.name: string = name
        self.version: string = version
        self.autor: string = autor

class TrainingConfig:
    def __init__(self, 
                n_epochs=5, 
                batch_size=16, 
                learning_rate=0.01, 
                loss_fn=None,
                patience=10,
                early_stop=True,
                model_save_path='model.pth'
                ):
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.learning_rate: float = learning_rate
        self.loss_fn = loss_fn 
        self.patience = patience
        self.early_stop: bool = early_stop
        self.model_save_path: str = model_save_path

    def __str__(self):
        return f"TrainingConfig(n_epochs={self.n_epochs}, batch_size={self.batch_size}, learning_rate={self.learning_rate}, loss_fn={self.loss_fn})"

piece_map = { # class 0 reserved for empty square
    "-": 0,
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
}

def fen_to_tensor(fen):
    '''
    Convert FEN string to tensor representation.
    Optimized using vectorized operations.
    example FEN: r1b2rk1/ppp2pbp/3q1np1/n3p1B1/2B5/1Q3N2/PP1N1PPP/3R1RK1 w - - 4 14 
    '''
    global piece_map
    fen_board = fen.split(' ')[0]
    
    # Expand FEN to 64 characters
    expanded = []
    for char in fen_board:
        if char.isdigit():
            expanded.extend(['-'] * int(char))
        elif char != '/':
            expanded.append(char)
    
    if len(expanded) != 64:
        raise ValueError(f"Invalid FEN: expanded to {len(expanded)} squares, expected 64")
    
    # Map to indices
    indices = [piece_map[c] for c in expanded]
    
    # Create one-hot tensor
    tensor = torch.zeros(64, len(piece_map))
    tensor[torch.arange(64), indices] = 1
    
    return tensor