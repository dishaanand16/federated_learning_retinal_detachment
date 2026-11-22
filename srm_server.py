"""Independent SRM site node.

SRM data is positive-only (every image is RD). It joins the central
aggregation server as an extra training client; it is never used for testing.

    python server.py --grpc        # central aggregation server (terminal 1)
    python srm_server.py           # SRM node                    (terminal 2)
"""
import argparse
from client import start_hospital_node
from srm_config import SRM_HOSPITAL, SRM_SERVER_ADDRESS


def main():
    parser = argparse.ArgumentParser(description="Start the SRM FL node")
    parser.add_argument("--server-address", default=SRM_SERVER_ADDRESS)
    args = parser.parse_args()
    start_hospital_node(SRM_HOSPITAL, args.server_address)


if __name__ == "__main__":
    main()
