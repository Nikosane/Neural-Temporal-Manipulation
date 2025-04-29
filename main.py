import argparse
import yaml
import torch
from models.transformer import TimeBenderTransformer
from data.lorenz import generate_lorenz_data
from utils.visualization import plot_trajectories


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def train(model, data, config):
    X, y = data
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), config['model_save_path'])


def infer(model, data, config):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    model = TimeBenderTransformer(config)

    X, y = generate_lorenz_data(config)
    train(model, (X, y), config)

    predictions = infer(model, X, config)
    plot_trajectories(X, y, predictions)


if __name__ == '__main__':
    main()
