# Running the FL pipeline

Use the `flenv` environment (`flenv\Scripts\python.exe`). Run everything from the project root.

## Sites

| Role | Hospital(s) | Config |
|---|---|---|
| Training clients | `LV_prasad`, `SMCH`, `Medanta` | `centralized_hospitals` |
| Training client (positive-only) | `srm` (all images are RD, label 1 assigned in code) | `srm_hospital`, `srm_config.py` |
| **Test hospital (never trains)** | `Aravid_eye_care` | `federated_hospital` (unchanged) |

`training_hospitals` in `config.py` is the client list both modes use. Set `INCLUDE_SRM = False` to reproduce the original 3-client experiment.

## Mode 1: simulation (unchanged command)

```
python server.py
```

## Mode 2: central server + independent hospital nodes (gRPC)

Each site runs as its own process and dials out to the central aggregation server on `SERVER_PORT` (default 8080). The server waits until all `len(training_hospitals)` nodes have connected.

```
python server.py --grpc                     # central aggregation server, evaluates on Aravid
python client.py --hospital LV_prasad
python client.py --hospital SMCH
python client.py --hospital Medanta
python srm_server.py                        # dedicated SRM node
```

Address overrides: env `FL_SERVER_PORT` / `FL_SERVER_ADDRESS` (all nodes), `SRM_SERVER_ADDRESS` (SRM node only), or `--server-address host:port` on any node.

## Tests

```
python -m unittest discover -s tests -v
```

## SRM notes

- SRM has no negatives. Its client therefore reports **sensitivity**, not AUC, because AUC is undefined with a single class. No negative samples are generated.
- With only positive labels, SRM's local updates push the model toward predicting RD. FedAvg weights it by its local batch count (4 of the 69 training batches). Watch Aravid specificity when comparing runs with and without SRM.
- Two SRM files are byte-identical (`WhatsApp Image 2026-10-22 at 2.38.38 PM.jpeg` and `... 2.38.41 PM (1).jpeg`). They can fall into different SRM local splits. This touches only SRM's own validation metric, not the Aravid test set.
