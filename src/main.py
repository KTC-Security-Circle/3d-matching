from utils.setup_logging import setup_logging


def main() -> None:
    logger = setup_logging(__name__)
    logger.info("Hello from 3d-matching!")


if __name__ == "__main__":
    main()
