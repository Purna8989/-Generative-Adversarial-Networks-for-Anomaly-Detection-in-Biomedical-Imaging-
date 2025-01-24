import argparse
from utils.data_loader import load_data
from models.f_anogan import FAnoGAN
from models.ganomaly import GANomaly
from models.multi_kd import MultiKD
from utils.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GAN models for anomaly detection")
    parser.add_argument('--model', type=str, required=True, choices=['f_anogan', 'ganomaly', 'multi_kd'], help="Model to train")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate")
    args = parser.parse_args()

    train_loader, test_loader = load_data(args.dataset, args.batch_size)

    if args.model == 'f_anogan':
        model = FAnoGAN(lr=args.lr)
    elif args.model == 'ganomaly':
        model = GANomaly(lr=args.lr)
    elif args.model == 'multi_kd':
        model = MultiKD(lr=args.lr)

    model.train(train_loader, epochs=args.epochs)
    results = model.test(test_loader)

    evaluate_model(results)

if __name__ == '__main__':
    main()
