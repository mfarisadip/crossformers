from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Implementasi MultiHeadAttention sesuai arsitektur transformer.

    menggunakan mekanisme perpecahan dimensi embedding ke beberapa head
    untuk memungkinkan model menangkap berbagai hubungan posisi

    Parameter:
        dimensi_embedding (int): dimensi vektor embedding (nilai default 512)
        heads (int): jumlah head attention yang digunakan (nilai default 8)

    Attribut:
        dimensi_embedding (int): menyimpan dimensi vektor embedding
        heads (int): menyimpan jumlah head
        head_dim (int): dimensi tiap head (dimensi_embedding // heads)
        query (nn.Linear): lapisan linear untuk proyeksi query (tanpa bias)
        key (nn.Linear): lapisan linear untuk proyeksi value (tanpa bias)
        fc_output (nn.Linear): lapisan linear untuk menggabungkan hasil head
        dropout (nn.Dropout): dropout untuk menambahkan noise pada attention scores

    Informasi tambahan:
        - mengamsumsikan dimensi_embedding dapat dibagi habis oleh heads
        - menggunakan einsum untuk operasi tensor yang efisien
        - mask bersifat opsional untuk skenario seperti padding atau lookahead

        - parameter bebas bias untuk proyeksi query / key / value
        - menggunakan dropout setelah proyeksi akhir (jika diaktifkan)
        - dapat menangani input dengan panjang sequence berbeda bentuk Q / K / V
        - kompatible dengan mask padding dan mask urutan (look-ahead)

    Forward parameter:
        key (torch.Tensor): tensor input key dengan bentuk (batch, k_len, dimensi_embedding)
        query (torch.Tensor): tensor input query dengan bentuk (batch, q_len, dimensi_embedding)
        value (torch.Tensor): tensor input value dengan bentuk (batch, v_len, dimensi_embedding)
        mask (torch.Tensor): tensor mask opsional dengan bentuk (batch, 1, q_len, k_len)

    Forward return:
        torch.Tensor: tensor output dengan bentuk (batch, q_len, dimensi_embedding)
    """

    def __init__(self, dimensi_embedding: int = 512, heads: int = 8, dropout: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert dimensi_embedding % heads == 0, f"Dimensi embedding {dimensi_embedding} tidak dapat dibagi habis dengan jumlah heads {heads}"
        self.dimensi_embedding = dimensi_embedding
        self.heads = heads
        self.head_dim = int(self.dimensi_embedding / self.heads)
        
        # Proyeksi pada dimensi penuh, bukan per-head
        self.query = nn.Linear(dimensi_embedding, dimensi_embedding, bias=False)
        self.key = nn.Linear(dimensi_embedding, dimensi_embedding, bias=False)
        self.value = nn.Linear(dimensi_embedding, dimensi_embedding, bias=False)
        
        self.fc_output = nn.Linear(dimensi_embedding, dimensi_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)
        
        # Proyeksi terlebih dahulu, kemudian reshape
        query = self.query(query).reshape(batch_size, q_len, self.heads, self.head_dim)
        key = self.key(key).reshape(batch_size, k_len, self.heads, self.head_dim)
        value = self.value(value).reshape(batch_size, v_len, self.heads, self.head_dim)
        
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / sqrt(self.head_dim)
        scores = F.softmax(product, dim=-1)
        scores = self.dropout(scores)  # Tambah dropout pada attention scores

        output = torch.einsum("bhqv,bvhd->bqhd", [scores, value]).reshape(
            batch_size, q_len, self.heads * self.head_dim
        )

        output = self.fc_output(output)
        return output


if __name__ == "__main__":
    dimensi_embedding: int = 512
    heads: int = 8
    attention_layer = MultiHeadAttention(dimensi_embedding, heads)

    batch_size: int = 32
    seq_len: int = 10
    key = torch.rand((batch_size, seq_len, dimensi_embedding))
    query = torch.rand((batch_size, seq_len, dimensi_embedding))
    value = torch.rand((batch_size, seq_len, dimensi_embedding))

    output = attention_layer(key, query, value)
    print(output.shape)