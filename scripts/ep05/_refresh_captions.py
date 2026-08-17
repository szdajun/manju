from pathlib import Path

src = Path('assets/characters/kid_dance/caption.txt')
dst_dir = Path('assets/characters/kid_dance/train_data/10_kid_dance')
text = src.read_text(encoding='utf-8').strip()
count = 0
for png in dst_dir.glob('*.png'):
    png.with_suffix('.txt').write_text(text, encoding='utf-8')
    count += 1
print(f'Wrote captions to {count} files')