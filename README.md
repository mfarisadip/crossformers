# CrossFormers
Implementasi transformer full-attention yang ringkas namun lengkap dengan berbagai fitur eksperimental relevan dari berbagai paper penelitian.

## Deskripsi

CrossFormers adalah implementasi arsitektur Transformer dengan mekanisme full-attention yang menggabungkan beberapa fitur eksperimental relevan dari berbagai paper penelitian terkini. Model ini dirancang untuk berbagai tugas NLP seperti machine translation, text summarization, dan language modeling.

## Arsitektur Model

Model CrossFormers terdiri dari beberapa komponen utama:

- **Encoder**: Memproses sequence input dan menghasilkan representasi kontekstual
  - Embedding layer untuk mengubah token menjadi vektor
  - Positional encoding untuk menambahkan informasi posisi
  - Stack dari N TransformerBlock untuk pemrosesan deep
  
- **Decoder**: Menghasilkan sequence output berdasarkan input encoder
  - Embedding dan positional encoding untuk token target 
  - Stack dari N DecoderBlock yang terdiri dari:
    - Masked multi-head self-attention
    - Cross-attention dengan output encoder
    - Feed-forward network
  
- **Attention**: Implementasi multi-head attention dengan operasi query, key, value
  - Mendukung masked attention untuk decoder

- **Normalization**: Mendukung LayerNorm standar dan implementasi custom Dynamic Tanh (DyT)

## Menjalankan Kode

Anda dapat menjalankan model dengan script `main.py` dan berbagai argumen:

```bash
python main.py --dimensi_embedding 512 --ukuran_vocab_sumber 10000 --ukuran_vocab_target 10000 --panjang_sekuens 100 --jumlah_block 6 --heads 8 --dropout 0.1 --norm_type layernorm
```

Atau Anda bisa menggunakan `uv` jika diinstall:

```bash
uv run main.py --dimensi_embedding 512 --ukuran_vocab_sumber 10000 --ukuran_vocab_target 10000 --panjang_sekuens 100 --jumlah_block 6 --heads 8 --dropout 0.1 --norm_type layernorm
```

### Parameter Penting

- `--dimensi_embedding`: Dimensi representasi embedding (default: 512)
- `--ukuran_vocab_sumber`: Ukuran kosakata input (default: 10000)
- `--ukuran_vocab_target`: Ukuran kosakata output (default: 10000)
- `--panjang_sekuens`: Panjang maksimum sequence (default: 100)
- `--jumlah_block`: Jumlah blok encoder/decoder (default: 6)
- `--faktor_ekspansi`: Faktor ekspansi untuk feed forward network (default: 4)
- `--heads`: Jumlah attention heads (default: 8)
- `--dropout`: Nilai dropout untuk regularisasi (default: 0.1)
- `--norm_type`: Jenis normalisasi, pilih 'layernorm' atau 'dyt' (default: 'layernorm')

## Training pada Dataset Enwik8

CrossFormer juga dilengkapi dengan script `train_enwik8.py` untuk melatih model pada dataset kompresi Enwik8:

```bash
python train_enwik8.py
```

Script ini akan:
- Melatih model pada 100.000 batch
- Menggunakan CrossFormer dengan Dynamic Tanh normalization
- Mengevaluasi pada validation set setiap 100 batch
- Menghasilkan sample text setiap 500 batch

## Fitur Khusus

### Dynamic Tanh (DyT) Normalization

CrossFormer mendukung layer normalisasi alternatif bernama Dynamic Tanh (DyT), yang dapat menjadi alternatif untuk LayerNorm standar. Untuk menggunakan DyT, atur parameter `norm_type` ke "dyt":

```bash
python main.py --norm_type dyt
```

DyT menggunakan fungsi aktivasi tanh dengan parameter yang dapat dilatih untuk meningkatkan fleksibilitas normalisasi.

## Requirement

Sebelum menjalankan proyek ini, pastikan Anda telah menginstal:
- PyTorch
- NumPy
- tqdm (untuk training progress bar)

## Referensi
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Introduction to Transformers](https://arxiv.org/abs/2304.10557)
- [Transformers without Normalization](https://arxiv.org/abs/2503.10622)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## Kredit
Kode ini merupakan modifikasi dan pengembangan lebih lanjut dari implementasi dasar Transformer oleh [WargaSlowy](https://github.com/WargaSlowy/transformer). Kami telah menambahkan fitur-fitur baru seperti Dynamic Tanh (DyT) normalization dan dukungan untuk dataset Enwik8.

## Ingin berkontribusi?

Kamu juga bisa berkontribusi di repositori ini, tapi dengan syarat kamu harus membaca [panduan dan pedoman kontribusi](CONTRIBUTING.md) ya.
