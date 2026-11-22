import argparse
import flwr as fl
import matplotlib.pyplot as plt
from model import build_model
from client import client_fn, load_data
from config import (
    NUM_ROUNDS, training_hospitals,
    federated_hospital, SERVER_PORT
)

round_log = []


def evaluate_fn(server_round, parameters, config):
    model  = build_model()
    val_ds = load_data(federated_hospital)[1]

    model.set_weights(parameters)
    loss, accuracy, auc = model.evaluate(val_ds, verbose=0)

    round_log.append({
        "round"   : server_round,
        "auc"     : round(auc, 4),
        "accuracy": round(accuracy, 4),
        "loss"    : round(loss, 4)
    })

    print(f"\n[SERVER] Round {server_round} — '{federated_hospital}'")
    print(f"         AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
    return loss, {"accuracy": accuracy, "auc": auc}


def print_summary(before_auc, before_acc, before_loss):
    after  = round_log[-1]
    sep    = "=" * 55

    print(f"\n{sep}")
    print("  ROUND-BY-ROUND RESULTS ON ARAVIND")
    print(sep)
    print(f"  {'Round':<8} {'AUC':>8} {'Accuracy':>10} {'Loss':>10}")
    print(f"  {'─'*38}")
    for r in round_log:
        print(f"  {r['round']:<8} {r['auc']:>8.4f} "
              f"{r['accuracy']:>10.4f} {r['loss']:>10.4f}")

    print(f"\n{sep}")
    print("  FINAL RESULTS SUMMARY")
    print(sep)
    print(f"  {'Metric':<12} {'Before FL':>10} {'After FL':>10} {'Change':>10}")
    print(f"  {'─'*44}")
    print(f"  {'AUC':<12} {before_auc:>10.4f} {after['auc']:>10.4f} "
          f"{after['auc']  - before_auc:>+10.4f}")
    print(f"  {'Accuracy':<12} {before_acc:>10.4f} {after['accuracy']:>10.4f} "
          f"{after['accuracy'] - before_acc:>+10.4f}")
    print(f"  {'Loss':<12} {before_loss:>10.4f} {after['loss']:>10.4f} "
          f"{after['loss'] - before_loss:>+10.4f}")
    print(sep)
    print(f"  Framework : Flower (flwr)")
    print(f"  Strategy  : FedAvg")
    print(f"  Rounds    : {NUM_ROUNDS}")
    print(f"  Clients   : {', '.join(training_hospitals)}")
    print(f"  New client: {federated_hospital}")
    print(f"  Note      : Raw data never left any hospital")
    print(sep)


def plot_results(before_auc, before_acc):
    rounds = [r["round"]    for r in round_log if r["round"] > 0]
    aucs   = [r["auc"]      for r in round_log if r["round"] > 0]
    accs   = [r["accuracy"] for r in round_log if r["round"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(rounds, aucs, marker='o', color='steelblue',
             linewidth=2, label='FL rounds')
    ax1.axhline(before_auc, color='tomato', linestyle='--',
                linewidth=1.5, label='Before FL')
    ax1.set_title(f"AUC per round — {federated_hospital}")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("AUC")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(rounds, accs, marker='s', color='seagreen',
             linewidth=2, label='FL rounds')
    ax2.axhline(before_acc, color='tomato', linestyle='--',
                linewidth=1.5, label='Before FL')
    ax2.set_title(f"Accuracy per round — {federated_hospital}")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fl_results.png", dpi=150)
    plt.show()
    print("\nPlot saved as fl_results.png")


def build_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = len(training_hospitals),
        min_evaluate_clients  = len(training_hospitals),
        min_available_clients = len(training_hospitals),
        evaluate_fn           = evaluate_fn,
    )


def evaluate_baseline():
    # Baseline — evaluate BEFORE FL on Aravind
    print("\nEvaluating baseline on Aravind BEFORE FL...")
    baseline_model         = build_model()
    aravind_val            = load_data(federated_hospital)[1]
    b_loss, b_acc, b_auc   = baseline_model.evaluate(aravind_val, verbose=1)
    print(f"Before FL — AUC: {b_auc:.4f} | Acc: {b_acc:.4f}")
    return b_loss, b_acc, b_auc


def main(grpc=False):
    print("=" * 55)
    print("  FLOWER FEDERATED LEARNING")
    print(f"  Clients  : {training_hospitals}")
    print(f"  Rounds   : {NUM_ROUNDS}")
    print(f"  Strategy : FedAvg")
    print("=" * 55)

    b_loss, b_acc, b_auc = evaluate_baseline()
    strategy             = build_strategy()
    server_config        = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)

    if grpc:
        # Central aggregation server; each hospital runs as its own node
        # (`python client.py --hospital <name>`, `python srm_server.py`)
        print(f"\n[SERVER] Waiting for {len(training_hospitals)} hospital nodes "
              f"on port {SERVER_PORT}...")
        fl.server.start_server(
            server_address = f"0.0.0.0:{SERVER_PORT}",
            config         = server_config,
            strategy       = strategy
        )
    else:
        # Run Flower simulation
        fl.simulation.start_simulation(
            client_fn        = client_fn,
            num_clients      = len(training_hospitals),
            config           = server_config,
            strategy         = strategy,
            client_resources = {"num_cpus": 2, "num_gpus": 0}
        )

    # Print summary and plot
    print_summary(b_auc, b_acc, b_loss)
    plot_results(b_auc, b_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL server")
    parser.add_argument("--grpc", action="store_true",
                        help="run as central aggregation server instead of a simulation")
    main(grpc=parser.parse_args().grpc)