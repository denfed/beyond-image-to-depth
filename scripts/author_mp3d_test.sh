# Run author's pretrained weights on Semantic-MP3D.

python3 test.py --img_path ~/datasets/mp3d-visualechoes/mp3d_split_wise_semantic/ \
                --audio_path ~/datasets/mp3d-visualechoes/echoes_navigable/ \
                --checkpoints_dir author_checkpoints/ \
                --dataset mp3d \
                --metadatapath dataset/metadata/mp3d