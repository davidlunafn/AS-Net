# AS-Net Hexagonal Architecture

This project follows the principles of hexagonal architecture (also known as ports and adapters) to create a loosely coupled and maintainable codebase.

## Directory Structure

*   **`domain`**: This is the core of the application. It contains the business logic and entities, and it is completely independent of any external technology or framework.
    *   `models`: Contains the domain models (e.g., `Audio`, `Spectrogram`).
    *   `services`: Contains the domain services, which orchestrate the business logic.

*   **`app`**: This layer contains the application-specific use cases. It acts as a bridge between the domain and the infrastructure.
    *   `services`: Contains the application services, which are responsible for handling the application's use cases.

*   **`adapters`**: This layer contains the adapters that connect the application to the outside world.
    *   `driven`: Contains the driven adapters, which are implementations of the ports defined in the application layer. These adapters are responsible for interacting with external systems, such as databases, file systems, and external APIs.
    *   `driving`: Contains the driving adapters, which are responsible for driving the application. These adapters can be a command-line interface (CLI), a web API, or a graphical user interface (GUI).
