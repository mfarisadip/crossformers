import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    """
    Dynamic Tanh (DyT) layer sebagai alternatif untuk LayerNorm.
    
    Parameter:
        num_features (int): Jumlah fitur (dimensi) input
        alpha_init_value (float): Nilai inisialisasi untuk parameter alpha (default: 0.5)
        
    Attribut:
        alpha (nn.Parameter): Parameter yang dapat dilatih untuk mengontrol kemiringan fungsi tanh
        weight (nn.Parameter): Parameter skala per-fitur
        bias (nn.Parameter): Parameter bias per-fitur
        
    Forward:
        x (torch.Tensor): Input tensor
        return (torch.Tensor): Output tensor setelah transformasi DyT
    """
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


def get_norm_layer(norm_type, dimensi_embedding):
    """
    Membuat layer normalisasi berdasarkan jenis yang ditentukan.
    
    Parameter:
        norm_type (str): Jenis normalisasi ('layernorm' atau 'dyt')
        dimensi_embedding (int): Dimensi embedding untuk normalisasi
        
    Return:
        nn.Module: Instance dari layer normalisasi yang dipilih
    """
    if norm_type.lower() == 'dyt':
        return DyT(dimensi_embedding)
    else:  # default: layernorm
        return nn.LayerNorm(dimensi_embedding)


def replikasi(block, N: int = 6) -> nn.ModuleList:
    """
    Membuat replikasi dari sebuah blok neural network sebanyak dari N kali

    Parameter:
        block (nn.Module): blok neural network yang akan direplikasi
                            harus merupakan instance dari `torch.nn.Module`
        N (int): jumlah replikasi yang diinginkan. default nilainya 6

    Return:
        nn.ModuleList: list dari blok neural network yang direplikasi
                        setiap elemen dalam list adalah salina mendalam dari
                        blok input
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack


class DPOLoss(nn.Module):
    """
    Implementasi Direct Preference Optimization (DPO) Loss.
    
    Parameters:
        beta (float): Parameter suhu yang mengontrol kekuatan regularisasi KL (default: 0.1)
        
    Forward:
        policy_chosen_logps (torch.Tensor): Log probabilitas dari model untuk respons pilihan
        policy_rejected_logps (torch.Tensor): Log probabilitas dari model untuk respons yang ditolak
        reference_chosen_logps (torch.Tensor): Log probabilitas dari model referensi untuk respons pilihan
        reference_rejected_logps (torch.Tensor): Log probabilitas dari model referensi untuk respons yang ditolak
        
    Return:
        torch.Tensor: Nilai loss DPO
    """
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
    
    def forward(
        self, 
        policy_chosen_logps, 
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps
    ):
        # Hitung KL divergence antara policy dan reference model
        pi_logratios = policy_chosen_logps - reference_chosen_logps
        pi_logratios_rejected = policy_rejected_logps - reference_rejected_logps
        
        # Hitung DPO loss
        logits = pi_logratios - pi_logratios_rejected
        losses = -F.logsigmoid(self.beta * logits)
        
        return losses.mean()


if __name__ == "__main__":

    class EncoderBlock(nn.Module):
        """
        Kelas representasi blok encoder dalam neural network

        Attribut:
            layer (nn.Linear): lapisan linear dengan input dan output ukuran 10
        """

        def __init__(self):
            super(EncoderBlock, self).__init__()
            self.layer = nn.Linear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Fungsi forward pass untuk block decoder

            Parameter:
                x (torch.Tensor): input tensor dengan dimensi sesuai dengan
                                    lapisan linear
            Return:
                torch.Tensor: output tensor hasil transformasi linear
            """
            return self.layer(x)

    # membuat instance dari EncoderBlock
    encoder_block = EncoderBlock()
    # membuat replikasi dari EncoderBlock sebanyak 6 kali
    encoder_stack = replikasi(encoder_block, N=6)
    # testing hasil replikasi
    print(encoder_stack)
