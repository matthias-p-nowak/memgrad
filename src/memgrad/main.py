import logging


def main():
    logging.info("Hello world")

def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    main()
