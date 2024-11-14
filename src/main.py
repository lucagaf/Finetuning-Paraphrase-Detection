import argparse
from  Training_Starter import Training_Starter


def main():
    parser = argparse.ArgumentParser(
        description="This Script starts a Trainings run fro the GLUETransformer Architecture"
    )

    # Necessary Arguements
    parser.add_argument("-wandb_key", required=True,
                        help="Provide your personal Weights & Biases API key here")
    parser.add_argument("-wandb_projectname", required=True,
                        help="Project name used in weights and biases.")

    # Optional Arguments
    parser.add_argument("-model_name", type=str, default="distilbert-base-uncased",
                        help="Please provide the model name to use for training. Default is 'distilbert-base-uncased'.")
    parser.add_argument("-task_name", type=str, default="mrpc",
                        help="Please provide the task name. Default is 'mrpc'.")
    # Add seed argument
    parser.add_argument("-seed", type=float, default=42,
                        help="Please provide the seed for random number generation. The default value is 42.")
    parser.add_argument("-lr", type=float, default=2e-5,
                        help="Please provide your desired Learning Rate which will be used for Training. The Default value is 2e-5")
    parser.add_argument("-warmup_steps", type=int, default=300,
                        help="Provide the your desired Amount of Warmup Steps. The default value is set to 300")
    parser.add_argument("-batch_size",  type=int, default=16,
                        help="Specify the training batch size. The default value is set to 16")
    parser.add_argument("-beta1", type=float, default=0.85,
                        help="The beta1 value is used for the Adam Optmizer. Please specify the float value for this parameter. The default value is set to 0.85")

    args = parser.parse_args()



    print("The training will start with the following values.")
    for arg, value in vars(args).items():
        if arg=="wandb_key":
            print(f"{arg}: ******")
        else:
            print(f"{arg}: {value}")


    training = Training_Starter(
        model_name=args.model_name,
        task_name=args.task_name,
        seed=args.seed,
        wandb_project_name=args.wandb_projectname,
        wandb_apikey=args.wandb_key,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
    )

    print("Training starts now")
    training.start_single_trainingsrun()


if __name__ == "__main__":
    main()


