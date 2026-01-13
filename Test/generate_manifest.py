import json
import os
import glob
import random

# 配置路径（根据你的实际调整）
data_root = '/proj/uppmax2025-3-5/ziqian/sabela/'  # 你的数据根目录
txt_dir = os.path.join(data_root, 'txt')
fon_dir = os.path.join(data_root, 'fon')
audio_dir = os.path.join(data_root, 'wav')  # 假设音频在这里
output_dir = '/proj/uppmax2025-3-5/ziqian/model/'  # 输出到test文件夹，便于脚本使用

# 定义相似/不同音（基于Galician vs Spanish phonology）
# Similar (简单学): 共享音，如元音/a e i o u/，辅音/p b t d k g f s m n l r ɾ j/ 等（~85%重叠）
# Different (狠狠学): Galician特有，如开中元音/ɛ ɔ/，/ʃ/ (x=sh)，gheada /ɰ ħ/，鼻化变体（这里用常见IPA符号；从fon提取实际）
similar_phonemes = {'a', 'e', 'i', 'o', 'u', 'p', 'b', 't', 'd', 'k', 'g', 'f', 's', 'm', 'n', 'l', 'r', 'ɾ', 'j', 'θ', 'tʃ', 'ɲ'}  # 扩展自标准inventory
sampa_to_ipa = {
    'E': 'ɛ', 'O': 'ɔ', 'x': 'ʃ', 'N': 'ŋ', 'tS': 'tʃ',
    'rr': 'r', 'r': 'ɾ', 'J': 'ɲ', 'T': 'θ', 
    'B': 'β', 'D': 'ð', 'G': 'ɣ', 'S': 's', 'Z': 'θ' # Z通常映射为清齿擦音
}
different_phonemes = {'ɛ', 'ɔ', 'ʃ', 'ɰ', 'ħ', 'ŋ', 'ʒ', 'ɟ'}  # Galician独特/变异（调整基于你的fon实际符号）

# 函数：从 .fon 提取 phoneme 序列（忽略时间戳）
def extract_phonemes(fon_path):
    phonemes = []
    try:
        with open(fon_path, 'r', encoding='latin-1') as f:  # 改成latin-1防编码错
            for line in f:
                parts = line.strip().split()  # 默认按空格分
                if len(parts) >= 2 and parts[0].replace('.', '').replace('#', '').isdigit():
                    p = parts[-1]  # 最后一个字段就是音素，如 'o', 'u'
                    if p == '#':   # 跳过 # 标记
                        continue
                    ipa_p = sampa_to_ipa.get(p, p)
                    phonemes.append(ipa_p)
    except Exception as e:
        print(f"Error reading {fon_path}: {e}")
    return ' '.join(phonemes)

# 函数：生成 manifest 条目
def generate_manifest_entries(file_ids, is_train=True):
    manifest = []
    for file_id in file_ids:
        txt_path = os.path.join(txt_dir, f'crpih_uvigo_gl_sabela_{file_id}.txt')
        fon_path = os.path.join(fon_dir, f'crpih_uvigo_gl_sabela_{file_id}.fon')
        audio_path = os.path.join(audio_dir, f'crpih_uvigo_gl_sabela_{file_id}.wav')
        
        if not all(os.path.exists(p) for p in [txt_path, fon_path, audio_path]):
            print(f"Skipping {file_id}: missing files")
            continue
        
        # 读 text（normalize: 小写，去除多余空格）
        try:
            with open(txt_path, 'r', encoding='latin-1') as f:  # 修复：用latin-1
                text = f.read().strip().lower()  # lower() 会把 Á → á 等，适合 TTS normalize
        except UnicodeDecodeError:
            print(f"Encoding error in {txt_path}, skipping")
            continue
        
        # 提取 phonemes
        phoneme_text = extract_phonemes(fon_path)
        
        entry = {
            "audio_filepath": audio_path,
            "text": text,  # grapheme 备份
            "phoneme_text": phoneme_text,  # 用于 phoneme mode（可选，在NeMo中可直接用g2p）
            "speaker": 'sabela',
            "duration": None  # NeMo会自动计算，跳过librosa以加速
        }
        
        # 实现“狠狠学”：如果含不同音，重复条目（仅train）
        phoneme_set = set(phoneme_text.split())
        if is_train and phoneme_set.intersection(different_phonemes):
            manifest.extend([entry] * 2)  # x2重复
        else:
            manifest.append(entry)
    
    output_file = os.path.join(output_dir, 'train_manifest.json' if is_train else 'val_manifest.json')
    with open(output_file, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + '\n')
    print(f"Generated {output_file} with {len(manifest)} entries")

# 主逻辑：找所有文件 ID（如 00001, 00002...，从fon提取并sort）
fon_files = sorted(glob.glob(os.path.join(fon_dir, '*.fon')))
file_ids = [os.path.basename(f).split('_')[-1].replace('.fon', '') for f in fon_files]

# Shuffle并split：90% train，10% val（随机更好）
random.shuffle(file_ids)
split_idx = int(0.9 * len(file_ids))
train_ids = file_ids[:split_idx]
val_ids = file_ids[split_idx:]

generate_manifest_entries(train_ids, is_train=True)
generate_manifest_entries(val_ids, is_train=False)

# 收集所有独特 phonemes（从所有fon）
all_phonemes = set()
for fon_file in fon_files:
    phonemes = extract_phonemes(fon_file).split()
    all_phonemes.update(phonemes)

phoneme_dict_path = os.path.join(output_dir, 'phoneme_dict_gl.txt')  # Galician版
with open(phoneme_dict_path, 'w', encoding='utf-8') as f:
    for i, p in enumerate(sorted(all_phonemes)):
        f.write(f"{p} {p}\n")  # NeMo g2p dict格式：grapheme phoneme（这里简单map自身，如果是符号表）

print(f"Phoneme dictionary saved to: {phoneme_dict_path} with {len(all_phonemes)} symbols")
print(f"Similar phonemes (for reference): {similar_phonemes}")
print(f"Different phonemes (for '狠狠学'): {different_phonemes}")
