"""Custom exceptions."""


class ServerNotRespondingException(Exception):
    """Exception raised if a server cannot be reached via ping.

    Attributes:
        ip_address -- ip_address of the server which caused the error
    """

    def __init__(self, ip_address: str) -> None:
        self.ip_address: str = ip_address
        self.message: str = f"MLFlow server at {self.ip_address} could not be reached! \n \
    Please check if you are connected to the MWN via a VPN..."
        super().__init__(self.message)


class InvalidCommandLineArgsException(Exception):
    """Exception raised if the program is called with a wrong number of command line arguments."""

    def __init__(self):
        self.message: str = """\
Not enough or invalid command line arguments supplied!

Program should be called like this: "python main.py AGENT_NAME EXPERIMENT_NAME YOUR_NAME
Required arguments:
    AGENT_NAME: the type of agent you want to use.
                Valid names are: random_edge, random_route, greedy, aco, ddqn, ppo
    EXPERIMENT_NAME: the MLFlow experiment that you want your runs to be logged as.
                If you are only testing, pass e.g. "test" as experiment name.
    YOUR_NAME: your own name (used for attributing who started a remote run)
        """
        super().__init__(self.message)


class InvalidAgentAlgorithm(Exception):
    """Exception raised if the user selects an invalid (unknown) agent algorithm."""

    def __init__(self):
        self.message: str = """\
Agent algorithm unknown.

Valid agent algorithms are: random_edge, random_route, greedy, aco, ddqn, ppo, coma.
        """
        super().__init__(self.message)
