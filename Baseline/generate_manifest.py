import json
import os
import glob
import random

data_root = '/data/'
txt_dir = os.path.join(data_root, 'txt')
fon_dir = os.path.join(data_root, 'fon')
audio_dir = os.path.join(data_root, 'wav')
output_dir = '/baseline/' 

similar_phonemes = {'a', 'e', 'i', 'o', 'u', 'p', 'b', 't', 'd', 'k', 'g', 'f', 's', 'm', 'n', 'l', 'r', 'ɾ', 'j', 'θ', 'tʃ', 'ɲ'} 
different_phonemes = {'ɛ', 'ɔ', 'ʃ', 'ɰ', 'ħ', 'ŋ', 'ʒ', 'ɟ'}
sampa_to_ipa = {
    'E': 'ɛ', 'O': 'ɔ', 'x': 'ʃ', 'N': 'ŋ', 'tS': 'tʃ',
    'rr': 'r', 'r': 'ɾ', 'J': 'ɲ', 'T': 'θ', 
    'B': 'β', 'D': 'ð', 'G': 'ɣ', 'S': 's', 'Z': 'θ'
}

def extract_phonemes(fon_path):
    phonemes = []
    try:
        with open(fon_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].replace('.', '').replace('#', '').isdigit():
                    p = parts[-1] 
                    if p == '#': 
                        continue
                    ipa_p = sampa_to_ipa.get(p, p)
                    phonemes.append(ipa_p)
    except Exception as e:
        print(f"Error reading {fon_path}: {e}")
    return ' '.join(phonemes)

def generate_manifest_entries(file_ids, is_train=True):
    manifest = []
    for file_id in file_ids:
        txt_path = os.path.join(txt_dir, f'crpih_uvigo_gl_sabela_{file_id}.txt')
        fon_path = os.path.join(fon_dir, f'crpih_uvigo_gl_sabela_{file_id}.fon')
        audio_path = os.path.join(audio_dir, f'crpih_uvigo_gl_sabela_{file_id}.wav')
        
        if not all(os.path.exists(p) for p in [txt_path, fon_path, audio_path]):
            print(f"Skipping {file_id}: missing files")
            continue
        
        try:
            with open(txt_path, 'r', encoding='latin-1') as f:
                text = f.read().strip().lower()
        except UnicodeDecodeError:
            print(f"Encoding error in {txt_path}, skipping")
            continue
        
        phoneme_text = extract_phonemes(fon_path)
        
        entry = {
            "audio_filepath": audio_path,
            "text": text,  
            "phoneme_text": phoneme_text, 
            "speaker": 'sabela',
            "duration": None
        }
        
        phoneme_set = set(phoneme_text.split())
        manifest.append(entry)
    
    output_file = os.path.join(output_dir, 'train_manifest.json' if is_train else 'val_manifest.json')
    with open(output_file, 'w') as f:
        for entry in manifest:
            f.write(json.dumps(entry) + '\n')
    print(f"Generated {output_file} with {len(manifest)} entries")

fon_files = sorted(glob.glob(os.path.join(fon_dir, '*.fon')))
file_ids = [os.path.basename(f).split('_')[-1].replace('.fon', '') for f in fon_files]

random.shuffle(file_ids)
split_idx = int(0.9 * len(file_ids))
train_ids = file_ids[:split_idx]
val_ids = file_ids[split_idx:]

generate_manifest_entries(train_ids, is_train=True)
generate_manifest_entries(val_ids, is_train=False)

all_phonemes = set()
for fon_file in fon_files:
    phonemes = extract_phonemes(fon_file).split()
    all_phonemes.update(phonemes)

phoneme_dict_path = os.path.join(output_dir, 'phoneme_dict_gl.txt') 
with open(phoneme_dict_path, 'w', encoding='utf-8') as f:
    for i, p in enumerate(sorted(all_phonemes)):
        f.write(f"{p} {p}\n")
        
print(f"Phoneme dictionary saved to: {phoneme_dict_path} with {len(all_phonemes)} symbols")
