import argparse
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.CNN_2D.UNet_L256 import UNet

from evaluation.eval_met import evaluate_all_subjects



def main():
    parser = argparse.ArgumentParser(description='Evaluate STNet model')
    parser.add_argument('--version', type=str)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing all subject directories')
    parser.add_argument('--dataName', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--STMap1', type=str, required=True, help='Name of the first STMap file')
    parser.add_argument('--STMap2', type=str, required=True, help='Name of the second STMap file')
    parser.add_argument('--frame_num', type=int, default=300, help='Number of frames per sample')
    parser.add_argument('--height', type=int, default=128, help='Number of height per sample')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--weights', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--save_dir', default=None, type=str, required=False, help='Directory to save Bland-Altman plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=args.channels, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights))
    print(f"Model loaded from {args.weights}")

    subject_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if
                    os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith('07')]

    print(subject_dirs)
    evaluate_all_subjects(subject_dirs, args.save_dir, model, device, args)

if __name__ == "__main__":
    main()

