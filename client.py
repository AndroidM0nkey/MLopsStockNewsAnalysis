import train from modules.train
import prepare_model, predict_sentiment from modules.infer

model = None

def main():
    parser = argparse.ArgumentParser(description="CLI menu for training and sentiment prediction.")
    parser.add_argument("mode", choices=["train_model", "predict_sentiment"], help="Choose the mode to run.")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.mode == "train_model":
        train_model()
    elif args.mode == "predict_sentiment":
        if model is None:
            model = prepare_model()
        print(f"Your label: {predict_sentiment(model)}")

if __name__ == "__main__":
    main()