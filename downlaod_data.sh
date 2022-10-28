mkdir sample_data
mkdir sample_data/meta
mkdir sample_data/features
mkdir -p results/all/statear/kin_poly/models_policy/ results/motion_im/uhc/models
gdown https://drive.google.com/uc?id=1W79g1It7VG6BWHbBZolbIkcWZsGlOCm9 -O  sample_data/
gdown https://drive.google.com/uc?id=1oFQ7aOxpGXmnizs76viV-z5-a32u95HB -O  sample_data/
gdown https://drive.google.com/uc?id=1GWndIhZZ91s6blqWccUzbPhHqrAeg7u1 -O  sample_data/features/ # Mocap annotations
gdown https://drive.google.com/uc?id=1JPbs9r5r5XYEFTecHHmrbgM-qHv0MKFl -O  sample_data/features/ # Mocap image features
gdown https://drive.google.com/uc?id=1qqKkWyIyT4rZyY91beRibjKqEB7iePXU -O  sample_data/meta/ # Mocap meta data
gdown https://drive.google.com/uc?id=1nkHGAnaMr-kog4XhIyLUcEEpxn2A1C4L -O  sample_data/features/ # real_world annotations
gdown https://drive.google.com/uc?id=1VxPliVe0dxAY1qGllnYxjbQsGlc4XYV8 -O  sample_data/meta/ # real world meta data
gdown https://drive.google.com/uc?id=1ZAHbM3iYe1Wq0ShTGeDpFhaZQAqhxHQb -O  sample_data/ # amass_occlusion
gdown https://drive.google.com/uc?id=1Hw2E8H0hHx9JwQXNsmWM0OjE1XTgFkmd -O results/motion_im/uhc/models/ # uhc trained model 
gdown https://drive.google.com/uc?id=1oQZzWVfWPrGzX0XyB0k4h7z6WLtSEsjX -O results/all/statear/kin_poly/models_policy/ # kin_poly trained model