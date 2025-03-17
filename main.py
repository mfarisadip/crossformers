import torch
import torch.nn as nn
import argparse
from crossformers.crossformers import CrossFormers

def parse_args():
    parser = argparse.ArgumentParser(description="Pelatihan dan evaluasi model CrossFormer")
    parser.add_argument("--dimensi_embedding", type=int, default=512, help="Dimensi embedding model")
    parser.add_argument("--ukuran_vocab_sumber", type=int, default=10000, help="Ukuran kosakata sumber")
    parser.add_argument("--ukuran_vocab_target", type=int, default=10000, help="Ukuran kosakata target")
    parser.add_argument("--panjang_sekuens", type=int, default=100, help="Panjang maksimum sekuens")
    parser.add_argument("--jumlah_block", type=int, default=6, help="Jumlah block encoder/decoder")
    parser.add_argument("--faktor_ekspansi", type=int, default=4, help="Faktor ekspansi FFN")
    parser.add_argument("--heads", type=int, default=8, help="Jumlah attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Nilai dropout")
    parser.add_argument("--batch_size", type=int, default=32, help="Ukuran batch untuk pelatihan")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch pelatihan")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "dyt"], 
                        help="Jenis normalisasi (layernorm atau dyt)")
    
    return parser.parse_args()

def buat_data_contoh(args):
    """Membuat data contoh untuk demonstrasi"""
    # Buat batch data contoh dengan ukuran (batch_size, panjang_sekuens)
    ukuran_vocab_kecil = min(10, args.ukuran_vocab_sumber) 
    sumber = torch.randint(0, ukuran_vocab_kecil, (args.batch_size, args.panjang_sekuens))
    
    # Buat target yang mirip dengan sumber tapi sedikit dimodifikasi
    # (misalnya shift 1 posisi atau flip beberapa bit)
    target = sumber.clone()
    
    return sumber, target

def main():
    args = parse_args()
    print("Memulai program CrossFormer...")
    
    # Inisialisasi model
    model = CrossFormers(
        dimensi_embedding=args.dimensi_embedding,
        ukuran_sumber_vocab=args.ukuran_vocab_sumber,
        ukuran_target_vocab=args.ukuran_vocab_target,
        panjang_sekuens=args.panjang_sekuens,
        jumlah_block=args.jumlah_block,
        faktor_ekspansi=args.faktor_ekspansi,
        heads=args.heads,
        dropout=args.dropout,
        norm_type=args.norm_type,
    )
    
    # Buat data contoh
    sumber, target = buat_data_contoh(args)
    print(f"Bentuk data sumber: {sumber.shape}")
    print(f"Bentuk data target: {target.shape}")
    
    # Jalankan forward pass
    try:
        outputs = model(sumber, target)
        print(f"Bentuk output model: {outputs.shape}")
        
        print("Model berhasil dijalankan!")
        print(model)
        
    except Exception as e:
        print(f"Terjadi kesalahan saat menjalankan model: {e}")
    
    # Tampilkan jumlah parameter model
    jumlah_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Jumlah parameter model: {jumlah_param:,}")
    print(f"Menggunakan normalisasi: {args.norm_type}")

if __name__ == "__main__":
    main()