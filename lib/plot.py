import matplotlib.pyplot as plt


# plotting training and validation losses
def plot_losses(train_losses, validation_losses, fig_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    # save the plot to a file if fig_path is provided
    if fig_path:
        plt.savefig(fig_path)
    plt.show()
