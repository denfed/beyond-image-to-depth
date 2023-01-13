# Run author's pretrained weights on Semantic-Replica.

python3 test.py --img_path ~/datasets/replica-visualechoes/scene_observations_semantic_128.pkl \
                --audio_path ~/datasets/replica-visualechoes/echoes_navigable/ \
                --checkpoints_dir author_checkpoints/ \
                --dataset replica